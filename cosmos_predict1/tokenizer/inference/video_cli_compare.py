# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Inference + side-by-side compare video with Token Mask visualization and Token Usage bar.
# Does NOT modify the original video_cli, video_lib, or infotok_inference.sh.
#
# Layout per frame:
#   +----------+----------+------------+-------------+
#   | Original | InfoTok  | Token Mask | Token Usage |
#   +----------+----------+------------+-------------+
#   |          |          | BW mask    | green bar   |
#   | orig vid | recon    | (NN up)    | block+clip  |
#   +----------+----------+------------+-------------+
#
# Usage: same args as video_cli; outputs *_compare.mp4.

import math
import os
import sys
from argparse import ArgumentParser, Namespace
from typing import Any

import cv2
import numpy as np
import torch
from loguru import logger as logging
from tqdm import tqdm

from cosmos_predict1.tokenizer.inference.utils import (
    get_filepaths,
    get_output_filepath,
    numpy2tensor,
    pad_video_batch,
    read_video,
    resize_video,
    tensor2numpy,
    unpad_video_batch,
    write_video,
)
from cosmos_predict1.tokenizer.inference.video_lib import CausalVideoTokenizer
from cosmos_predict1.tokenizer.networks import TokenizerConfigs

# ── Visual constants ──────────────────────────────────────────────
BAR_COLOR_RGB = (52, 168, 83)
BAR_COL_WIDTH = 140
BAR_INNER_W = 32
CAPTION_HEIGHT = 40
CAPTION_BG_RGB = (30, 30, 30)
CAPTION_FG_BGR = (255, 255, 255)
CLIP_LINE_BGR = (180, 180, 180)
CAPTION_FONT = cv2.FONT_HERSHEY_SIMPLEX
CAPTION_FONT_SCALE = 0.7
CAPTION_THICKNESS = 2
BAR_FONT_SCALE = 0.5
BAR_FONT_THICKNESS = 1


# ── Arg parsing ───────────────────────────────────────────────────
def _parse_args() -> Namespace:
    p = ArgumentParser(description="InfoTok compare: original | recon | mask | token usage.")
    p.add_argument("--video_pattern", type=str, default="path/to/videos/*.mp4")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--checkpoint_enc", type=str, default=None)
    p.add_argument("--checkpoint_dec", type=str, default=None)
    p.add_argument(
        "--tokenizer_type", type=str, default=None,
        choices=[
            "CV8x8x8-720p", "DV8x16x16-720p", "CV4x8x8-360p", "DV4x8x8-360p",
            "OURS4x8x8-256p", "OURS4x8x8-256p-88", "OURS4x8x8-mse-256p-88",
            "OURS4x8x8-order4-256p-88", "OURS4x8x8-concat-256p-88", "OURS4x8x8-special-256p-88",
        ],
    )
    p.add_argument("--mode", type=str, choices=["torch", "jit"], default="jit")
    p.add_argument("--short_size", type=int, default=None)
    p.add_argument("--temporal_window", type=int, default=17)
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--strategy", type=str, default="static")
    p.add_argument("--avg_rate", type=float, default=0.5)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--output_fps", type=float, default=24.0)
    p.add_argument("--overlap_window", type=int, default=0)
    return p.parse_args()


# ── Core: inference with mask capture ─────────────────────────────
@torch.no_grad()
def run_inference_with_mask(
    autoencoder: CausalVideoTokenizer,
    batch_video: np.ndarray,
    temporal_window: int,
    overlap_window: int,
    strategy: str,
    avg_rate: float,
):
    """Replicate forward / forward_with_overlap, but also capture per-window binary masks.

    Returns:
        output_video   : np.ndarray [B, T, H, W, 3] uint8
        all_masks      : list of np.ndarray [T_tok, H_tok, W_tok] bool (True = kept)
        all_block_rates: list of np.ndarray [T_tok] float  (actual kept ratio per block)
        all_clip_rates : list of float  (target rate per window)
        window_ranges  : list of (start_frame, end_frame)
        temporal_compression: int
        token_spatial_hw: (H_tok, W_tok)
    """
    num_frames = batch_video.shape[1]
    use_overlap = overlap_window > 0
    step_size = temporal_window - overlap_window if use_overlap else temporal_window
    if use_overlap:
        iters = math.ceil((num_frames - temporal_window) / step_size) + 1 if num_frames > temporal_window else 1
    else:
        iters = (num_frames - 1) // temporal_window + 1

    model = autoencoder._full_model
    tc = model.temporal_compression
    has_method = hasattr(model, "method")

    all_outputs: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []
    all_block_rates: list[np.ndarray] = []
    all_clip_rates: list[float] = []
    window_ranges: list[tuple[int, int]] = []

    for idx in tqdm(range(iters), desc="inference+mask"):
        start = 0 if (idx == 0 or not use_overlap) else idx * step_size
        if not use_overlap:
            start = idx * temporal_window
        end = min(start + temporal_window, num_frames)

        inp = batch_video[:, start:end, ...]
        padded, crop = pad_video_batch(inp)
        tensor_in = numpy2tensor(padded, dtype=autoencoder._dtype, device=autoencoder._device)

        hidden = model.encode(tensor_in)[1]
        B, D, T_tok, H_tok, W_tok = hidden.shape

        if has_method:
            mask_seq = (model.method == "mse") or getattr(model, "special_attn", False)
            chunk_loss = autoencoder.compute_chunk_loss(hidden, tensor_in)

            if strategy == "global_elbo":
                rate = avg_rate * torch.mean(chunk_loss) / autoencoder.elbo_mean
                rate = rate.clip(0.0625, 1.0).to(hidden.device)
                use_strat = "elbo"
            elif strategy == "global_elbo_bin":
                rate = avg_rate * torch.mean(chunk_loss) / autoencoder.elbo_mean
                rate = 0.25 * round(rate.item() / 0.25)
                rate = min(max(rate, 0.0625), 1.0)
                rate = torch.tensor([rate]).to(hidden.device)
                use_strat = "elbo"
            else:
                rate = torch.tensor(avg_rate).to(hidden.device)
                use_strat = strategy

            alloc = model.get_allocated_ratios(
                hidden, use_adaptive=True, chunk_loss=chunk_loss,
                manual_base_rate=rate, mask_seq=mask_seq,
                overwrite_strategy=use_strat, rescale=False,
            )
            masked_h, adaptive_mask = model.mask_tokens(
                hidden, alloc, mask_method=model.method, chunk_loss=chunk_loss,
            )

            if mask_seq:
                mask_3d = adaptive_mask[0, 0, 0].cpu().bool().numpy().reshape(T_tok, H_tok, W_tok)
            else:
                mask_3d = adaptive_mask[0, 0].cpu().bool().numpy().reshape(T_tok, H_tok, W_tok)

            clip_rate = alloc.float().mean().item()
        else:
            masked_h = hidden
            mask_3d = np.ones((T_tok, H_tok, W_tok), dtype=bool)
            clip_rate = 1.0

        out_tensor = model.decode(masked_h)
        out_np = unpad_video_batch(tensor2numpy(out_tensor), crop)

        block_rates = np.array([mask_3d[t].sum() / (H_tok * W_tok) for t in range(T_tok)])

        all_outputs.append(out_np)
        all_masks.append(mask_3d)
        all_block_rates.append(block_rates)
        all_clip_rates.append(clip_rate)
        window_ranges.append((start, end))

    # ── Blend overlapping windows ─────────────────────────────────
    output_video = all_outputs[0]
    for idx in range(1, len(all_outputs)):
        win = all_outputs[idx]
        ov = min(overlap_window, output_video.shape[1], win.shape[1]) if use_overlap else 0
        if ov > 0:
            prev = output_video[:, -ov:].astype(np.float32)
            nxt = win[:, :ov].astype(np.float32)
            a = np.array([(ov - i) / (ov + 1) for i in range(ov)], dtype=np.float32).reshape(1, -1, 1, 1, 1)
            blended = np.clip(a * prev + (1 - a) * nxt, 0, 255).astype(np.uint8)
            output_video = np.concatenate([output_video[:, :-ov], blended, win[:, ov:]], axis=1)
        else:
            output_video = np.concatenate([output_video, win], axis=1)

    return (output_video, all_masks, all_block_rates, all_clip_rates,
            window_ranges, tc, (H_tok, W_tok))


# ── Per-frame lookups ─────────────────────────────────────────────
def _frame_to_window(f: int, window_ranges: list[tuple[int, int]]) -> int:
    """Return the index of the earliest window that contains frame f."""
    for idx, (s, e) in enumerate(window_ranges):
        if s <= f < e:
            return idx
    return len(window_ranges) - 1


def get_frame_info(
    f: int,
    tc: int,
    all_masks: list[np.ndarray],
    all_block_rates: list[np.ndarray],
    all_clip_rates: list[float],
    window_ranges: list[tuple[int, int]],
):
    """Return (mask_2d, block_rate, clip_rate) for global frame f."""
    wi = _frame_to_window(f, window_ranges)
    start = window_ranges[wi][0]
    local_f = f - start
    token_t = max(0, (local_f - 1) // tc) if local_f > 0 else 0
    token_t = min(token_t, all_masks[wi].shape[0] - 1)
    return all_masks[wi][token_t], all_block_rates[wi][token_t], all_clip_rates[wi]


# ── Drawing helpers ───────────────────────────────────────────────
def _center_text_x(text, rx, rw, font, scale, thickness):
    (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
    return rx + max(0, (rw - tw) // 2)


def _build_caption_row(W: int, col_w: int) -> np.ndarray:
    total_w = W * 3 + col_w
    row = np.full((CAPTION_HEIGHT, total_w, 3), CAPTION_BG_RGB, dtype=np.uint8)
    bgr = np.ascontiguousarray(row[:, :, ::-1])
    ty = CAPTION_HEIGHT - 12
    for label, rx, rw in [
        ("Original", 0, W), ("InfoTok", W, W),
        ("Token Mask", 2 * W, W), ("Token Usage", 3 * W, col_w),
    ]:
        tx = _center_text_x(label, rx, rw, CAPTION_FONT, CAPTION_FONT_SCALE, CAPTION_THICKNESS)
        cv2.putText(bgr, label, (tx, ty), CAPTION_FONT, CAPTION_FONT_SCALE,
                    CAPTION_FG_BGR, CAPTION_THICKNESS, cv2.LINE_AA)
    return np.ascontiguousarray(bgr[:, :, ::-1])


def _build_mask_frame(mask_2d: np.ndarray, H: int, W: int) -> np.ndarray:
    """Nearest-neighbor upscale a bool mask to H×W RGB (white=kept, black=masked)."""
    gray = (mask_2d.astype(np.uint8) * 255)
    rgb = np.stack([gray, gray, gray], axis=-1)
    return cv2.resize(rgb, (W, H), interpolation=cv2.INTER_NEAREST)


def _build_bar_column(H: int, col_w: int, block_pct: float) -> np.ndarray:
    """Token Usage column: green bar = per-block rate."""
    col = np.full((H, col_w, 3), CAPTION_BG_RGB, dtype=np.uint8)
    bx0 = (col_w - BAR_INNER_W) // 2
    bx1 = bx0 + BAR_INNER_W

    bh = int((block_pct / 100.0) * H)
    yt = max(0, H - bh)
    col[yt:H, bx0:bx1, 0] = BAR_COLOR_RGB[0]
    col[yt:H, bx0:bx1, 1] = BAR_COLOR_RGB[1]
    col[yt:H, bx0:bx1, 2] = BAR_COLOR_RGB[2]

    bgr = np.ascontiguousarray(col[:, :, ::-1])

    bt = f"{int(round(block_pct))}%"
    btx = _center_text_x(bt, 0, col_w, CAPTION_FONT, BAR_FONT_SCALE, BAR_FONT_THICKNESS)
    bty = max(14, yt - 6)
    cv2.putText(bgr, bt, (btx, bty), CAPTION_FONT, BAR_FONT_SCALE,
                CAPTION_FG_BGR, BAR_FONT_THICKNESS, cv2.LINE_AA)

    return np.ascontiguousarray(bgr[:, :, ::-1])


# ── Frame composition ─────────────────────────────────────────────
def compose_compare_video(
    original: np.ndarray,
    reconstructed: np.ndarray,
    all_masks, all_block_rates, all_clip_rates,
    window_ranges, tc,
) -> np.ndarray:
    """Build [caption] / [original | recon | mask | bar] for every frame."""
    T, H, W, C = original.shape
    col_w = BAR_COL_WIDTH
    total_w = W * 3 + col_w
    total_h = CAPTION_HEIGHT + H
    caption = _build_caption_row(W, col_w)
    out = np.zeros((T, total_h, total_w, C), dtype=np.uint8)

    for f in tqdm(range(T), desc="composing"):
        mask_2d, block_rate, clip_rate = get_frame_info(
            f, tc, all_masks, all_block_rates, all_clip_rates, window_ranges,
        )
        mask_vis = _build_mask_frame(mask_2d, H, W)
        bar = _build_bar_column(H, col_w, block_rate * 100.0)
        row = np.concatenate([original[f], reconstructed[f], mask_vis, bar], axis=1)
        out[f] = np.concatenate([caption, row], axis=0)
    return out


# ── Main ──────────────────────────────────────────────────────────
args = _parse_args()
if args.mode == "torch" and args.tokenizer_type is None:
    logging.error("`torch` backend requires `--tokenizer_type`.")
    sys.exit(1)


def _run_compare() -> None:
    if args.checkpoint is None and args.checkpoint_enc is None and args.checkpoint_dec is None:
        logging.warning("Need checkpoint (or enc+dec).")
        return

    _config = TokenizerConfigs[args.tokenizer_type.replace("-", "_")].value if args.mode == "torch" else None
    logging.info(f"Loading model from {os.path.dirname(args.checkpoint or args.checkpoint_enc or args.checkpoint_dec)} ...")
    autoencoder = CausalVideoTokenizer(
        checkpoint=args.checkpoint,
        checkpoint_enc=args.checkpoint_enc,
        checkpoint_dec=args.checkpoint_dec,
        tokenizer_config=_config,
        device=args.device,
        dtype=args.dtype,
    )

    filepaths = get_filepaths(args.video_pattern)
    logging.info(f"Found {len(filepaths)} videos.")

    # ── Global ELBO collection pass (if needed) ───────────────────
    if "global_elbo" in args.strategy:
        for filepath in filepaths:
            video = read_video(filepath)
            video = resize_video(video, short_size=args.short_size)
            bv = video[np.newaxis, ...]
            _ = autoencoder(bv, temporal_window=args.temporal_window,
                            strategy=args.strategy, avg_rate=args.avg_rate,
                            collect_elbo_only=True, overlap_window=args.overlap_window)
        if len(autoencoder.elbos) > 0:
            et = torch.stack(autoencoder.elbos) if isinstance(autoencoder.elbos[0], torch.Tensor) else torch.tensor(autoencoder.elbos)
            autoencoder.elbo_mean = et.mean().item()

    # ── Per-video inference + compose ─────────────────────────────
    for filepath in filepaths:
        logging.info(f"Processing {filepath} ...")
        video = read_video(filepath)
        video = resize_video(video, short_size=args.short_size)
        batch_video = video[np.newaxis, ...]

        (output_video, all_masks, all_block_rates, all_clip_rates,
         window_ranges, tc, _) = run_inference_with_mask(
            autoencoder, batch_video, args.temporal_window,
            args.overlap_window, args.strategy, args.avg_rate,
        )
        recon = output_video[0]

        compare = compose_compare_video(
            video, recon,
            all_masks, all_block_rates, all_clip_rates,
            window_ranges, tc,
        )

        out_path = get_output_filepath(filepath, output_dir=args.output_dir)
        base, ext = os.path.splitext(out_path)
        compare_path = f"{base}_compare{ext}"
        logging.info(f"Writing {compare_path} ...")
        write_video(compare_path, compare, fps=args.output_fps)


@logging.catch(reraise=True)
def main() -> None:
    _run_compare()


if __name__ == "__main__":
    main()
