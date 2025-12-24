# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A CLI to run CausalVideoTokenizer on plain videos based on torch.jit.

Usage:
    python3 -m cosmos_predict1.tokenizer.inference.video_cli \
        --video_pattern 'path/to/video/samples/*.mp4' \
        --output_dir ./reconstructions \
        --checkpoint_enc ./checkpoints/<model-name>/encoder.jit \
        --checkpoint_dec ./checkpoints/<model-name>/decoder.jit

    Optionally, you can run the model in pure PyTorch mode:
    python3 -m cosmos_predict1.tokenizer.inference.video_cli \
        --video_pattern 'path/to/video/samples/*.mp4' \
        --mode=torch \
        --tokenizer_type=CV \
        --temporal_compression=4 \
        --spatial_compression=8 \
        --checkpoint_enc ./checkpoints/<model-name>/encoder.jit \
        --checkpoint_dec ./checkpoints/<model-name>/decoder.jit
"""

import os
import sys
from argparse import ArgumentParser, Namespace
from typing import Any

import numpy as np
from loguru import logger as logging
from tqdm import tqdm

from cosmos_predict1.tokenizer.inference.utils import (
    get_filepaths,
    get_output_filepath,
    read_video,
    resize_video,
    write_video,
)
from cosmos_predict1.tokenizer.inference.video_lib import CausalVideoTokenizer
from cosmos_predict1.tokenizer.networks import TokenizerConfigs


def _parse_args() -> tuple[Namespace, dict[str, Any]]:
    parser = ArgumentParser(description="A CLI for CausalVideoTokenizer.")
    parser.add_argument(
        "--video_pattern",
        type=str,
        default="path/to/videos/*.mp4",
        help="Glob pattern.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="JIT full Autoencoder model filepath.",
    )
    parser.add_argument(
        "--checkpoint_enc",
        type=str,
        default=None,
        help="JIT Encoder model filepath.",
    )
    parser.add_argument(
        "--checkpoint_dec",
        type=str,
        default=None,
        help="JIT Decoder model filepath.",
    )
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        default=None,
        choices=[
            "CV8x8x8-720p",
            "DV8x16x16-720p",
            "CV4x8x8-360p",
            "DV4x8x8-360p",
            "OURS4x8x8-256p",
            "OURS4x8x8-256p-88",
            "OURS4x8x8-mse-256p-88",
            "OURS4x8x8-order4-256p-88",
            "OURS4x8x8-concat-256p-88",
            "OURS4x8x8-special-256p-88",
        ],
        help="Specifies the tokenizer type.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["torch", "jit"],
        default="jit",
        help="Specify the backend: native 'torch' or 'jit' (default: 'jit')",
    )
    parser.add_argument(
        "--short_size",
        type=int,
        default=None,
        help="The size to resample inputs. None, by default.",
    )
    parser.add_argument(
        "--temporal_window",
        type=int,
        default=17,
        help="The temporal window to operate at a time.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Sets the precision, default bfloat16.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for invoking the model.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="static",
        help="Strategy for invoking the model.",
    )
    parser.add_argument(
        "--avg_rate",
        type=float,
        default=0.5,
        help="Average rate for invoking the model.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory.")
    parser.add_argument(
        "--output_fps",
        type=float,
        default=24.0,
        help="Output frames-per-second (FPS).",
    )
    parser.add_argument(
        "--save_input",
        action="store_true",
        help="If on, the input video will be be outputted too.",
    )
    parser.add_argument(
        "--save_clip",
        action="store_true",
        help="If on, the input video will be be outputted too.",
    )
    parser.add_argument(
        "--only_square_clips",
        action="store_true",
        help="If on, only square clips will be saved.",
    )
    parser.add_argument(
        "--overlap_window",
        type=int,
        default=0,
        help="If using overlapping during inference",
    )
    args = parser.parse_args()
    return args


logging.info("Initializes args ...")
args = _parse_args()
if args.mode == "torch" and args.tokenizer_type is None:
    logging.error("`torch` backend requires `--tokenizer_type` to be specified.")
    sys.exit(1)


def _run_eval() -> None:
    """Invokes JIT-compiled CausalVideoTokenizer on an input video."""

    if args.checkpoint_enc is None and args.checkpoint_dec is None and args.checkpoint is None:
        logging.warning("Aborting. Both encoder or decoder JIT required. Or provide the full autoencoder JIT model.")
        return

    if args.mode == "torch":
        _type = args.tokenizer_type.replace("-", "_")
        _config = TokenizerConfigs[_type].value
    else:
        _config = None

    logging.info(
        f"Loading a torch.jit model `{os.path.dirname(args.checkpoint or args.checkpoint_enc or args.checkpoint_dec)}` ..."
    )
    autoencoder = CausalVideoTokenizer(
        checkpoint=args.checkpoint,
        checkpoint_enc=args.checkpoint_enc,
        checkpoint_dec=args.checkpoint_dec,
        tokenizer_config=_config,
        device=args.device,
        dtype=args.dtype,
    )

    logging.info(f"Looking for files matching video_pattern={args.video_pattern} ...")
    filepaths = get_filepaths(args.video_pattern)
    logging.info(f"Found {len(filepaths)} videos from {args.video_pattern}.")

    print("Tokenizer Type: ", args.tokenizer_type)

    if args.save_clip:
        for filepath in filepaths:
            logging.info(f"Reading video {filepath} ...")
            video = read_video(filepath)
            output_filepath = get_output_filepath(filepath, output_dir=args.output_dir)
            video = resize_video(video, short_size=args.short_size)
            logging.info("Splitting video into clips ...")
            num_frames = video.shape[0]
            print("num_frames: ", num_frames)
            print("video.shape: ", video.shape)
            temporal_window = args.temporal_window
            for idx in tqdm(range(0, (num_frames - 1) // temporal_window + 1)):
                # Input video for the current window.
                start, end = idx * temporal_window, (idx + 1) * temporal_window
                video_clip = video[start:end, ...]
                print("video_clip.shape: ", video_clip.shape)
                if video_clip.shape[0] < temporal_window:
                    continue
                if args.only_square_clips:
                    if video_clip.shape[1] == video_clip.shape[2]: # only save square clips
                        clip_file_path = output_filepath.replace(".mp4", f"_{idx}.mp4")
                        write_video(clip_file_path, video_clip, fps=args.output_fps)
                else:
                    clip_file_path = output_filepath.replace(".mp4", f"_{idx}.mp4")
                    write_video(clip_file_path, video_clip, fps=args.output_fps)
        return

    if "global_elbo" in args.strategy:
        for filepath in filepaths:
            logging.info(f"Reading video {filepath} ...")
            video = read_video(filepath)
            video = resize_video(video, short_size=args.short_size)

            logging.info("Invoking the autoencoder model in ... ")
            batch_video = video[np.newaxis, ...]
            _ = autoencoder(batch_video, temporal_window=args.temporal_window, strategy=args.strategy, avg_rate=args.avg_rate, collect_elbo_only=True, overlap_window=args.overlap_window)
        
        # Compute ELBO mean from collected values
        import torch
        if len(autoencoder.elbos) > 0:
            elbos_tensor = torch.stack(autoencoder.elbos) if isinstance(autoencoder.elbos[0], torch.Tensor) else torch.tensor(autoencoder.elbos)
            autoencoder.elbo_mean = elbos_tensor.mean().item()

    for filepath in filepaths:
        logging.info(f"Reading video {filepath} ...")
        video = read_video(filepath)
        video = resize_video(video, short_size=args.short_size)

        logging.info("Invoking the autoencoder model ...")
        batch_video = video[np.newaxis, ...]
        output_video, _ = autoencoder(batch_video, temporal_window=args.temporal_window, strategy=args.strategy, avg_rate=args.avg_rate, overlap_window=args.overlap_window)
        output_video = output_video[0]
        
        output_filepath = get_output_filepath(filepath, output_dir=args.output_dir)
        logging.info(f"Outputing {output_filepath} ...")
        write_video(output_filepath, output_video, fps=args.output_fps)
        if args.save_input:
            ext = os.path.splitext(output_filepath)[-1]
            input_filepath = output_filepath.replace(ext, "_input" + ext)
            write_video(input_filepath, video, fps=args.output_fps)

@logging.catch(reraise=True)
def main() -> None:
    _run_eval()


if __name__ == "__main__":
    main()