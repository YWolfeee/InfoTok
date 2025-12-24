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

"""A library for Causal Video Tokenizer inference."""

import math
from typing import Any
from einops import rearrange

import numpy as np
import torch
from tqdm import tqdm

from cosmos_predict1.tokenizer.inference.utils import (
    load_decoder_model,
    load_encoder_model,
    load_model,
    numpy2tensor,
    pad_video_batch,
    tensor2numpy,
    unpad_video_batch,
)


class CausalVideoTokenizer(torch.nn.Module):
    def __init__(
        self,
        checkpoint: str = None,
        checkpoint_enc: str = None,
        checkpoint_dec: str = None,
        tokenizer_config: dict[str, Any] = None,
        device: str = "cuda",
        dtype: str = "bfloat16",
    ) -> None:
        super().__init__()
        self._device = device
        self._dtype = getattr(torch, dtype)
        self._full_model = (
            load_model(checkpoint, tokenizer_config, device).to(self._dtype) if checkpoint is not None else None
        )
        self._enc_model = (
            load_encoder_model(checkpoint_enc, tokenizer_config, device).to(self._dtype)
            if checkpoint_enc is not None
            else None
        )
        self._dec_model = (
            load_decoder_model(checkpoint_dec, tokenizer_config, device).to(self._dtype)
            if checkpoint_dec is not None
            else None
        )
    
    @torch.no_grad()
    def collect_video_elbo(self, input_tensor: torch.Tensor, elbo_base: float = 1.0, overlap_window: int = 0, is_start: bool = False) -> torch.Tensor:
        """Collect ELBO (Evidence Lower Bound) loss for each temporal block.
        
        Args:
            input_tensor: The input tensor Bx3xTxHxW layout, range [-1..1].
        Returns:
            The ELBO loss for each temporal block.
        """
        # Check if self has elbos attribute, if not, create a list
        if not hasattr(self, 'elbos'):
            self.elbos = []
        if self._full_model is not None:
            hidden_tensor = self._full_model.encode(input_tensor)[1] # output tensor: [1, 6, 13, 32, 52]
            block_loss = self.compute_chunk_loss(hidden_tensor, input_tensor)
            temporal_compression = (input_tensor.shape[2] - 1) // (hidden_tensor.shape[2] - 1)
            start_idx = 0 if is_start else (overlap_window + temporal_compression - 1) // temporal_compression
            video_elbo = torch.mean(block_loss[:, start_idx:])
            self.elbos.append(video_elbo)
        else:
            raise NotImplementedError("collect_elbo is not implemented for AdaptiveVideoTokenizer")
        
    @torch.no_grad()
    def compute_chunk_loss(self, hidden_tensor: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        recon_full = self._full_model.decode(hidden_tensor)
        full_loss = (input_tensor - recon_full).abs().mean(dim=1)
        chunk_loss = full_loss[:, 1:].reshape(full_loss.shape[0], -1, self._full_model.temporal_compression, *full_loss.shape[2:]).mean(dim=2)
        chunk_loss = torch.concat([full_loss[:, :1], chunk_loss], dim=1)
        chunk_loss = rearrange(chunk_loss, "b t (h u) (w v) -> b t h u w v", u=self._full_model.spatial_compression, v=self._full_model.spatial_compression).mean(dim=(-1,-3))
        return chunk_loss

    @torch.no_grad()
    def autoencode(self, input_tensor: torch.Tensor, strategy: str = "static", avg_rate: float = 0.5, return_encode: bool = False) -> torch.Tensor:
        """Reconstrcuts a batch of video tensors after embedding into a latent.

        Args:
            video: The input video Bx3xTxHxW layout, range [-1..1].
        Returns:
            The reconstructed video, layout Bx3xTxHxW, range [-1..1].
        """
        if self._full_model is not None:
            # input tensor: [1, 3, 33, 256, 416]
            hidden_tensor = self._full_model.encode(input_tensor)[1] # output tensor: [1, 6, 9, 32, 52]
            if hasattr(self._full_model, 'method'):
                if self._full_model.method == "mse" or self._full_model.special_attn:
                    mask_seq = True
                else:
                    mask_seq = False
                chunk_loss = self.compute_chunk_loss(hidden_tensor, input_tensor)
                if strategy == "global_elbo":
                    rate = avg_rate * torch.mean(chunk_loss) / self.elbo_mean
                    rate = rate.clip(0.0625, 1.0).to(hidden_tensor.device)
                    use_strategy = "elbo"
                elif strategy == 'global_elbo_bin':
                    rate = avg_rate * torch.mean(chunk_loss) / self.elbo_mean
                    rate = 0.25 * round(rate.item() / 0.25)
                    rate = min(max(rate, 0.0625), 1.0)
                    rate = torch.tensor([rate]).to(hidden_tensor.device)
                    use_strategy = "elbo"

                else:
                    rate = torch.tensor(avg_rate).to(hidden_tensor.device)
                    use_strategy = strategy
                allocated_ratios = self._full_model.get_allocated_ratios(hidden_tensor, use_adaptive=True, chunk_loss=chunk_loss, manual_base_rate=rate, mask_seq=mask_seq, overwrite_strategy=use_strategy, rescale=False)
                print("Current allocated ratios: ", allocated_ratios)
                hidden_tensor, _ = self._full_model.mask_tokens(hidden_tensor, allocated_ratios, mask_method=self._full_model.method, chunk_loss=chunk_loss)
            else:
                allocated_ratios = torch.ones_like(hidden_tensor[:, 0, :, 0, 0])
            
            if return_encode:
                return hidden_tensor, allocated_ratios
            output_tensor = self._full_model.decode(hidden_tensor)
        else:
            output_latent = self.encode(input_tensor)[0]
            output_tensor = self.decode(output_latent)
        
        return output_tensor, allocated_ratios
    
    @torch.no_grad()
    def encode(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor]:
        """Encodes a numpy video into a CausalVideo latent or code.

        Args:
            input_tensor: The input tensor Bx3xTxHxW layout, range [-1..1].
        Returns:
            For causal continuous video (CV) tokenizer, the tuple contains:
                - The latent embedding, Bx16x(t)x(h)x(w), where the compression
                rate is (T/t x H/h x W/w), and channel dimension of 16.
            For causal discrete video (DV) tokenizer, the tuple contains:
              1) The indices, Bx(t)x(h)x(w), from a codebook of size 64K, which
                is formed by FSQ levels of (8,8,8,5,5,5).
              2) The discrete code, Bx6x(t)x(h)x(w), where the compression rate
                is again (T/t x H/h x W/w), and channel dimension of 6.
        """
        assert input_tensor.ndim == 5, "input video should be of 5D."

        output_latent = self._enc_model(input_tensor)
        if isinstance(output_latent, torch.Tensor):
            return output_latent
        return output_latent[:-1]

    @torch.no_grad()
    def decode(self, input_latent: torch.Tensor) -> torch.Tensor:
        """Encodes a numpy video into a CausalVideo latent.

        Args:
            input_latent: The continuous latent Bx16xtxhxw for CV,
                        or the discrete indices Bxtxhxw for DV.
        Returns:
            The reconstructed tensor, layout [B,3,1+(T-1)*8,H*16,W*16] in range [-1..1].
        """
        assert input_latent.ndim >= 4, "input latent should be of 5D for continuous and 4D for discrete."
        return self._dec_model(input_latent)

    def forward(
        self,
        video: np.ndarray,
        temporal_window: int = 17,
        strategy: str = "static",
        avg_rate: float = 0.5,
        collect_elbo_only: bool = False,
        overlap_window: int = 0,
    ) -> np.ndarray:
        """Reconstructs video using a pre-trained CausalTokenizer autoencoder.
        Given a video of arbitrary length, the forward invokes the CausalVideoTokenizer
        in a sliding manner with a `temporal_window` size.

        Args:
            video: The input video BxTxHxWx3 layout, range [0..255].
            temporal_window: The length of the temporal window to process, default=25.
            overlap_window: Number of overlapping frames between consecutive windows.
        Returns:
            The reconstructed video in range [0..255], layout BxTxHxWx3.
        """
        assert video.ndim == 5, "input video should be of 5D."
        
        if overlap_window > 0:
            return self.forward_with_overlap(video, temporal_window, strategy, avg_rate, collect_elbo_only, overlap_window)
        
        num_frames = video.shape[1]  # can be of any length.
        output_video_list = []

        if "video_elbo" in strategy:
            raise NotImplementedError("video_elbo is bugged and not fully tested yet.")

        if "video_elbo" in strategy or ("global_elbo" in strategy and collect_elbo_only):
            print("Start collecting global ELBO ...")
            if strategy != "video_elbo" and "global_elbo" not in strategy:
                elbo_base = float(strategy[:3]) # [1.0, 0.5]
            else:
                elbo_base = 1.0
            for idx in tqdm(range(0, (num_frames - 1) // temporal_window + 1)):
                # Input video for the current window.
                start, end = idx * temporal_window, (idx + 1) * temporal_window
                input_video = video[:, start:end, ...]

                # Spatio-temporally pad input_video so it's evenly divisible.
                padded_input_video, crop_region = pad_video_batch(input_video)
                input_tensor = numpy2tensor(padded_input_video, dtype=self._dtype, device=self._device)
                self.collect_video_elbo(input_tensor, elbo_base)
            print("Collected ELBO: ", self.elbos[-1])
        
        if collect_elbo_only:
            return

        for idx in tqdm(range(0, (num_frames - 1) // temporal_window + 1)):
            # Input video for the current window.
            start, end = idx * temporal_window, (idx + 1) * temporal_window
            input_video = video[:, start:end, ...]

            # Spatio-temporally pad input_video so it's evenly divisible.
            padded_input_video, crop_region = pad_video_batch(input_video)
            input_tensor = numpy2tensor(padded_input_video, dtype=self._dtype, device=self._device)
            # self.get_token_loss_curve(input_tensor)
            # assert False
            output_tensor, token_rate = self.autoencode(input_tensor, strategy=strategy, avg_rate=avg_rate)
            padded_output_video = tensor2numpy(output_tensor)
            output_video = unpad_video_batch(padded_output_video, crop_region)

            output_video_list.append(output_video)
        
        # Convert token_rate from torch.tensor to list of float if needed
        if isinstance(token_rate, torch.Tensor):
            token_rate = token_rate.detach().cpu().tolist()
        return np.concatenate(output_video_list, axis=1), token_rate
    
    @torch.no_grad()
    def forward_with_overlap(
        self,
        video: np.ndarray,
        temporal_window: int = 17,
        strategy: str = "static",
        avg_rate: float = 0.5,
        collect_elbo_only: bool = False,
        temporal_overlap: int = 1,
    ) -> np.ndarray:
        """Reconstructs video using a pre-trained CausalTokenizer autoencoder with overlap blending.
        
        The overlapping frames between consecutive windows are blended using linear interpolation:
        - For overlap=3: frame weights are [0.75, 0.5, 0.25] for previous window, [0.25, 0.5, 0.75] for next window
        - For overlap=5: frame weights are [5/6, 4/6, 3/6, 2/6, 1/6] for previous window
        
        Window positioning example (temporal_window=17, overlap=3):
        - Window 1: frames [0, 17)
        - Window 2: frames [17 - 3, 17 - 3 + 17) = [14, 31)
        - Overlapping frames: 14, 15, 16

        Args:
            video: The input video BxTxHxWx3 layout, range [0..255].
            temporal_window: The length of the temporal window to process, default=17.
            temporal_overlap: Number of overlapping frames between consecutive windows.
        Returns:
            The reconstructed video in range [0..255], layout BxTxHxWx3.
            Token rates for each window (concatenated).
        """
        assert video.ndim == 5, "input video should be of 5D."
        assert temporal_overlap < temporal_window, "temporal_overlap must be less than temporal_window"

        if "video_elbo" in strategy:
            raise NotImplementedError("video_elbo is bugged and not fully tested yet.")

        num_frames = video.shape[1]
        step_size = temporal_window - temporal_overlap
        
        # Calculate number of iterations
        if num_frames <= temporal_window:
            iters = 1
        else:
            iters = math.ceil((num_frames - temporal_window) / step_size) + 1

        # Handle ELBO collection if needed
        if "video_elbo" in strategy or ("global_elbo" in strategy and collect_elbo_only):
            print("Start collecting global ELBO ...")
            for idx in tqdm(range(0, iters)):
                start = 0 if idx == 0 else idx * step_size
                end = min(start + temporal_window, num_frames)
                input_video = video[:, start:end, ...]
                padded_input_video, crop_region = pad_video_batch(input_video)
                input_tensor = numpy2tensor(padded_input_video, dtype=self._dtype, device=self._device)
                self.collect_video_elbo(input_tensor, temporal_overlap, is_start=(idx == 0))
        
        if collect_elbo_only:
            return
        
        # Per-window autoencode with pixel-level alpha blending
        # Each window is independently encoded and decoded, then overlapping frames are blended.
        all_token_rates = []
        output_video = None
        
        for idx in tqdm(range(0, iters)):
            start = 0 if idx == 0 else idx * step_size
            end = min(start + temporal_window, num_frames)
            
            # Process current window
            input_video_window = video[:, start:end, ...]
            padded_input_video, crop_region = pad_video_batch(input_video_window)
            input_tensor = numpy2tensor(padded_input_video, dtype=self._dtype, device=self._device)
            output_tensor, token_rate = self.autoencode(input_tensor, strategy=strategy, avg_rate=avg_rate)
            padded_output_video = tensor2numpy(output_tensor)
            output_video_window = unpad_video_batch(padded_output_video, crop_region)
            
            all_token_rates.append(token_rate)
            
            if output_video is None:
                # First window, no blending needed
                output_video = output_video_window
            else:
                # Blend overlapping region
                actual_overlap = min(temporal_overlap, output_video.shape[1], output_video_window.shape[1])
                
                if actual_overlap > 0:
                    # Get overlap regions and convert to float32 for blending
                    # output_video shape: [B, T, H, W, 3], dtype: uint8
                    overlap_region_prev = output_video[:, -actual_overlap:, ...].astype(np.float32)
                    overlap_region_next = output_video_window[:, :actual_overlap, ...].astype(np.float32)
                    
                    # Create alpha weights for blending
                    # alpha[i] = (actual_overlap - i) / (actual_overlap + 1) for prev window
                    # Example: overlap=3 -> alphas = [0.75, 0.5, 0.25]
                    # Example: overlap=5 -> alphas = [5/6, 4/6, 3/6, 2/6, 1/6]
                    alphas = np.array([(actual_overlap - i) / (actual_overlap + 1) 
                                       for i in range(actual_overlap)], dtype=np.float32)
                    # Reshape for broadcasting: [1, overlap, 1, 1, 1]
                    alphas = alphas.reshape(1, -1, 1, 1, 1)
                    
                    # Blend: alpha * prev + (1 - alpha) * next
                    blended_overlap = alphas * overlap_region_prev + (1 - alphas) * overlap_region_next
                    blended_overlap = np.clip(blended_overlap, 0, 255).astype(np.uint8)
                    
                    # Combine: non-overlap part of prev + blended overlap + non-overlap part of next
                    output_video = np.concatenate([
                        output_video[:, :-actual_overlap, ...],
                        blended_overlap,
                        output_video_window[:, actual_overlap:, ...]
                    ], axis=1)
                else:
                    # No overlap, just concatenate
                    output_video = np.concatenate([output_video, output_video_window], axis=1)
        
        # Concatenate all token rates
        # token_rate shape: [B, T_tokens] where T_tokens is the number of temporal tokens
        encoded_token_rates = torch.cat(all_token_rates, dim=1)
        
        return output_video, encoded_token_rates
    
class AdaptiveVideoTokenizer(torch.nn.Module):
    # TODO: Implement the inference code of AdaptiveVideoTokenizer
    def __init__(self, **kwargs):
        super().__init__()
        pass

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        pass