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

from hydra.core.config_store import ConfigStore

from cosmos_predict1.utils import log
from cosmos_predict1.utils.lazy_config import LazyDict
from cosmos_predict1.tokenizer.training.configs.experiments.utils import create_debug_job_with_mock_data

# -------------------------------------------------
# Hyperparameters of experiments on adaptive tokenization of CUSTOM videos
# Matches OURS4x8x8-mse-256p-88 inference config
# -------------------------------------------------

NUM_VIDEO_FRAMES = 33 # this should be temporal_compression_rate * n + 1 (n=1,2,...)
CROP_HEIGHT = 256 # This indicates the largest width / height of the video
TEMPORAL_COMPRESSION = 4
SPATIAL_COMPRESSION = 8
MAX_NUM_TOKENS = (NUM_VIDEO_FRAMES - 1) // TEMPORAL_COMPRESSION * (CROP_HEIGHT // SPATIAL_COMPRESSION) ** 2
MIN_NUM_TOKENS = MAX_NUM_TOKENS // 16

# ------------ ViT backbone config (OURS4x8x8-mse-256p-88) ------------

ADV4x8x8_256p_CUSTOM_Posttrain: LazyDict = LazyDict(
    dict(
        defaults=[
            "/experiment/video_basic",
            {"override /network": "adaptive_discrete_video"},
            {"override /data_train": "custom_video256"},
            {"override /data_val": "custom_video256"},
            {"override /scheduler": "warmup_cosine"},
            "_self_",
        ],
        dataloader_train=dict(
            dataset=dict(
                crop_height=256,
                num_video_frames=33,
            ),
            batch_size=1,
        ),
        dataloader_val=dict(
            dataset=dict(
                crop_height=256,
                num_video_frames=33,
            ),
            batch_size=1,
        ),
        model=dict(
            config=dict(
                network=dict(
                    # Matches ours_discrete_video_4x8x8_mse_256p config from configs.py
                    quantizer="FSQ",
                    embedding_dim=6,
                    levels=[8, 8, 8, 5, 5, 5],
                    patch_size=2,
                    patch_method="haar",
                    z_channels=256,
                    legacy_mode=False,
                    temporal_compression=4,
                    spatial_compression=8,
                    # ViT backbone configuration
                    vit_config=dict(
                        hidden_size=256,
                        intermediate_size=512,
                        num_encoder_layers=8,
                        num_decoder_layers=8,
                        num_attention_heads=32,
                        theta=10000.0,
                        rms_norm_eps=1e-5,
                        initializer_range=0.02,
                        max_num_video_frames=49,
                        switch_rotary_to_1d=1.0,
                        max_sequence_length=96,
                        max_sequence_length_1d=8192,
                        concat_decode_2d=False,
                        special_attn=False,
                        method="mse",
                    ),
                    # Adaptive tokenization settings
                    rate_strategy="elbo",
                    use_vit=True,
                    freeze_original=False,
                )
            )
        ),
        job=dict(
            project="infotok_posttraining",
            group="tokenizer",
            name="OURS4x8x8-mse-256p-88_Posttrain",
        ),
        checkpoint=dict(
            # Set to False because Cosmos checkpoint doesn't have ViT weights
            # The base encoder/decoder will load, ViT will be randomly initialized
            strict_resume=False,
            load_training_state=False,
            jit=dict(input_shape=[1, 3, 33, 256, 256]),
        ),
        scheduler=dict(
            warmup_iters=5000,
            lr_decay_iters=100000,
            min_lr=1e-5,
        ),

    )
)

cs = ConfigStore.instance()

for _item in [
    ADV4x8x8_256p_CUSTOM_Posttrain,  # Register this for custom video training
]:
    experiment_name = [name for name, value in globals().items() if value is _item][0]

    log.info(f"Registering experiment: {experiment_name}")
    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )

    mock_experiment = f"mock_{experiment_name}"
    log.info(f"Registering mock experiment: {mock_experiment}")
    _debug_item = create_debug_job_with_mock_data(_item["job"]["name"])
    cs.store(
        group="experiment",
        package="_global_",
        name=mock_experiment,
        node=_debug_item,
    )

