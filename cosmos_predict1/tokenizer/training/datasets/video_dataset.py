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
PYTHONPATH=. python cosmos_predict1/tokenizer/training/datasets/video_dataset.py

Adapted from:
https://github.com/bytedance/IRASim/blob/main/dataset/dataset_3D.py
"""

import os
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm

from cosmos_predict1.diffusion.training.datasets.dataset_utils import ToTensorVideo


class Dataset(Dataset):
    def __init__(
        self,
        video_pattern,
        sequence_interval=1,
        start_frame_interval=1,
        num_video_frames=25,
    ):
        """
        Dataset class for loading image-text-to-video generation data.
        
        Args:
            video_directory_or_pattern (str): path/to/files/*.mp4 or *.jpeg, etc.  # NOTICE: Updated docstring to include image files
            sequence_interval (int): Interval between sampled frames in a sequence.
            num_video_frames (int): Number of frames to load per sequence.
        Returns dict with:
            - video: RGB frames tensor [C,T,H,W] (T=1 for images if sequence_length is 1)  # NOTICE: Updated return format description
            - video_name: Dict with metadata.
        """
        super().__init__()
        self.video_directory_or_pattern = video_pattern
        self.start_frame_interval = start_frame_interval
        self.sequence_interval = sequence_interval
        self.sequence_length = num_video_frames

        # NOTICE: Added file extension sets to distinguish between videos and images
        self.video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
        self.image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

        self.video_paths = sorted(glob(str(video_pattern)))
        print(f"{len(self.video_paths)} videos in total")

        self.samples = self._init_samples(self.video_paths)
        self.samples = sorted(self.samples, key=lambda x: (x["video_path"], x["frame_ids"][0]))
        print(f"{len(self.samples)} samples in total")
        self.wrong_number = 0
        self.preprocess = T.Compose(
            [
                ToTensorVideo(),
            ]
        )

    def __str__(self):
        return f"{len(self.video_paths)} samples from {self.video_directory_or_pattern}"

    # NOTICE: Added helper methods to identify file types
    def _is_video_file(self, path):
        ext = os.path.splitext(path)[1].lower()
        return ext in self.video_extensions

    def _is_image_file(self, path):
        ext = os.path.splitext(path)[1].lower()
        return ext in self.image_extensions

    def _init_samples(self, video_paths):
        samples = []
        with ThreadPoolExecutor(32) as executor:
            future_to_video_path = {  # NOTICE: Renamed variable and method
                executor.submit(self._load_and_process_video_path, video_path): video_path for video_path in video_paths
            }
            for future in tqdm(as_completed(future_to_video_path), total=len(video_paths)):
                samples.extend(future.result())
        return samples

    # NOTICE: Renamed and updated method to handle both videos and images
    def _load_and_process_video_path(self, video_path):
        samples = []
        if self._is_video_file(video_path):
            # Use decord.VideoReader for videos.
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
            n_frames = len(vr)
            for frame_i in range(0, n_frames, self.start_frame_interval):
                sample = dict()
                sample["video_path"] = video_path
                sample["frame_ids"] = []
                curr_frame_i = frame_i
                while True:
                    if curr_frame_i > (n_frames - 1):
                        break
                    sample["frame_ids"].append(curr_frame_i)
                    if len(sample["frame_ids"]) == self.sequence_length:
                        break
                    curr_frame_i += self.sequence_interval
                if len(sample["frame_ids"]) == self.sequence_length:
                    samples.append(sample)
        elif self._is_image_file(video_path):  # NOTICE: Added handling for image files
            # For images, treat as 1-frame video.
            sample = dict()
            sample["video_path"] = video_path
            # For image, we can either use a single frame or repeat it.
            # Here, we simply use a single frame.
            sample["frame_ids"] = [0]
            samples.append(sample)
        else:
            warnings.warn(f"File {video_path} is neither a recognized video nor image. Skipped.")
        return samples

    def _load_video(self, video_path, frame_ids):
        # Decord VideoReader for videos
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        assert (np.array(frame_ids) < len(vr)).all()
        assert (np.array(frame_ids) >= 0).all()
        vr.seek(0)
        frame_data = vr.get_batch(frame_ids).asnumpy()
        return frame_data

    # NOTICE: Added method to load images
    def _load_image(self, image_path):
        # Use PIL to load image
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            frame = np.array(img)
        return frame

    # NOTICE: Updated to handle both videos and images
    def _get_frames(self, video_path, frame_ids):
        if self._is_image_file(video_path):
            # Load image and create a "video" with one frame.
            frame = self._load_image(video_path)  # shape: (H, W, C)
            frame = frame.astype(np.uint8)
            # Convert to tensor and add a time dimension.
            frame = torch.from_numpy(frame).permute(2, 0, 1)  # [C, H, W]
            assert self.sequence_length == 1, "under this setting, sequence_length must be 1"
            frame = frame.unsqueeze(0)  # [T, C, H, W]
        else:
            # Load frames from video.
            frames = self._load_video(video_path, frame_ids)
            frames = frames.astype(np.uint8)
            # Decord returns [T, H, W, C]; convert to [T, C, H, W]
            frame = torch.from_numpy(frames).permute(0, 3, 1, 2)
        # Apply preprocessing and convert to uint8.
        frame = self.preprocess(frame)
        frame = torch.clamp(frame * 255.0, 0, 255).to(torch.uint8)
        return frame.contiguous() # NOTICE: add this to avoid non_resizable error when loading data with batch_size > 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        try:
            sample = self.samples[index]
            video_path = sample["video_path"]
            frame_ids = sample["frame_ids"]
            data = dict()

            frames = self._get_frames(video_path, frame_ids)
            # Rearrange frames from [T, C, H, W] to [C, T, H, W]
            video = frames.permute(1, 0, 2, 3)
            data["is_image"] = video.shape[1] == 1
            data["video"] = video
            data["video_name"] = {
                "video_path": video_path,
                "start_frame_id": str(frame_ids[0]),
            }
            data["fps"] = 24
            # Example image_size; adjust as needed.
            data["image_size"] = torch.tensor([704, 1280, 704, 1280]) #.cuda()   # TODO: Does this matter?
            data["num_frames"] = self.sequence_length
            data["padding_mask"] = torch.zeros(1, 704, 1280) #.cuda()
            return data
        except Exception:
            warnings.warn(
                f"Invalid data encountered: {self.samples[index]['video_path']}. Skipped "
                f"(by randomly sampling another sample in the same dataset)."
            )
            warnings.warn("FULL TRACEBACK:")
            warnings.warn(traceback.format_exc())
            self.wrong_number += 1
            print(self.wrong_number)
            return self[np.random.randint(len(self.samples))]


if __name__ == "__main__":
    dataset = Dataset(
        video_directory_or_pattern="assets/example_training_data/videos/*.mp4",
        sequence_interval=1,
        num_frames=57,
        video_size=[240, 360],
    )

    indices = [0, 13, 200, -1]
    for idx in indices:
        data = dataset[idx]
        print((f"{idx=} " f"{data['video'].sum()=}\n" f"{data['video'].shape=}\n" f"{data['video_name']=}\n" "---"))

