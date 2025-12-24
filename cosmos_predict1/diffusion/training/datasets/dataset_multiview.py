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

"""
Run this command to interactively debug:
PYTHONPATH=. python cosmos_predict1/diffusion/training/datasets/dataset_multiview.py

Adapted from:
https://github.com/bytedance/IRASim/blob/main/dataset/dataset_3D.py
"""

import os
import pickle
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm

from cosmos_predict1.diffusion.training.datasets.dataset_utils import Resize_Preprocess, ToTensorVideo


class MockDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dir,
        num_frames,
        num_views,
        video_size,
    ):
        """
        A mock dataset that generates synthetic video, text embeddings, and metadata
        for testing purposes.

        Args:
            dataset_dir (str): Path to the dataset directory (not used in mock data).
            num_frames (int): Number of frames to load per view.
            num_views (int): Number of views per sequence.
            video_size (list): Target size [H,W] for video frames
        """

        super().__init__()
        self.sequence_length = num_frames
        self.num_views = num_views
        self.video_size = video_size  # (height, width)
        self.video_paths = ["mock_video_path"] * 10_000  # Simulated dataset size

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, index: int) -> dict:
        total_frames = self.sequence_length * self.num_views  # Compute total frames
        height, width = self.video_size

        data = {
            "video": torch.randint(0, 255, (3, total_frames, height, width), dtype=torch.uint8),
            "video_name": {
                "video_path": "mock_video_path",
                "t5_embedding_path": "mock_t5_embedding_path",
                "start_frame_id": "0",
            },
            "t5_text_embeddings": torch.randn(512 * self.num_views, 1024),
            "t5_text_mask": torch.ones(512, dtype=torch.int64),
            "fps": 24,
            "image_size": torch.tensor([height, width, height, width]),  # [H, W, H, W]
            "num_frames": 57,
            "padding_mask": torch.zeros(1, height, width),
        }

        return data


if __name__ == "__main__":
    dataset = Dataset(
        dataset_dir="assets/example_training_data/",
        num_frames=57,
        num_views=6,
        video_size=[240, 360],
    )

    indices = [0, 13, 200, -1]
    for idx in indices:
        data = dataset[idx]
        print(
            (
                f"{idx=} "
                f"{data['video'].sum()=}\n"
                f"{data['video'].shape=}\n"
                f"{data['video_name']=}\n"
                f"{data['t5_text_embeddings'].shape=}\n"
                "---"
            )
        )
