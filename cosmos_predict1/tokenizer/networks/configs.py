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

"""The default image and video tokenizer configs."""

from cosmos_predict1.tokenizer.modules import (
    ContinuousFormulation,
    Decoder3DType,
    DecoderType,
    DiscreteQuantizer,
    Encoder3DType,
    EncoderType
)

continuous_image = dict(
    # The attention resolution for res blocks.
    attn_resolutions=[32],
    # The base number of channels.
    channels=128,
    # The channel multipler for each resolution.
    channels_mult=[2, 4, 4],
    dropout=0.0,
    in_channels=3,
    # The spatial compression ratio.
    spatial_compression=16,
    # The number of layers in each res block.
    num_res_blocks=2,
    out_channels=3,
    resolution=1024,
    patch_size=4,
    patch_method="haar",
    # The output latent dimension (channels).
    latent_channels=16,
    # The encoder output channels just before sampling.
    # Which is also the decoder's input channels.
    z_channels=16,
    # A factor over the z_channels, to get the total channels the encoder should output.
    # For a VAE for instance, we want to output the mean and variance, so we need 2 * z_channels.
    z_factor=1,
    name="CI",
    # What formulation to use, either "AE" or "VAE".
    # Chose VAE here, since the pre-trained ckpt were of a VAE formulation.
    formulation=ContinuousFormulation.AE.name,
    # Specify type of encoder ["Default", "LiteVAE"]
    encoder=EncoderType.Default.name,
    # Specify type of decoder ["Default"]
    decoder=DecoderType.Default.name,
)
continuous_image_8x8_360p = dict(continuous_image)
continuous_image_8x8_360p["patch_size"] = 2
continuous_image_8x8_360p["spatial_compression"] = 8

continuous_image_16x16_360p = dict(continuous_image)
continuous_image_16x16_360p["patch_size"] = 2
continuous_image_16x16_360p["spatial_compression"] = 16


discrete_image = dict(
    # The attention resolution for res blocks.
    attn_resolutions=[32],
    # The base number of channels.
    channels=128,
    # The channel multipler for each resolution.
    channels_mult=[2, 4, 4],
    dropout=0.0,
    in_channels=3,
    # The spatial compression ratio.
    spatial_compression=16,
    # The number of layers in each res block.
    num_res_blocks=2,
    out_channels=3,
    resolution=1024,
    patch_size=4,
    patch_method="haar",
    # The encoder output channels just before sampling.
    z_channels=256,
    # A factor over the z_channels, to get the total channels the encoder should output.
    # for discrete tokenization, often we directly use the vector, so z_factor=1.
    z_factor=1,
    # The quantizer of choice, VQ, LFQ, FSQ, or ResFSQ.
    quantizer=DiscreteQuantizer.FSQ.name,
    # The embedding dimension post-quantization, which is also the input channels of the decoder.
    # Which is also the output
    embedding_dim=6,
    # The number of levels to use for fine-scalar quantization.
    levels=[8, 8, 8, 5, 5, 5],
    # The number of quantizers to use for residual fine-scalar quantization.
    num_quantizers=4,
    name="DI",
    # Specify type of encoder ["Default", "LiteVAE"]
    encoder=EncoderType.Default.name,
    # Specify type of decoder ["Default"]
    decoder=DecoderType.Default.name,
)
discrete_image_8x8_360p = dict(discrete_image)
discrete_image_8x8_360p["patch_size"] = 2
discrete_image_8x8_360p["spatial_compression"] = 8

discrete_image_16x16_360p = dict(discrete_image)
discrete_image_16x16_360p["patch_size"] = 2
discrete_image_16x16_360p["spatial_compression"] = 16

continuous_video = dict(
    attn_resolutions=[32],
    channels=128,
    channels_mult=[2, 4, 4],
    dropout=0.0,
    in_channels=3,
    num_res_blocks=2,
    out_channels=3,
    resolution=1024,
    patch_size=4,
    patch_method="haar",
    latent_channels=16,
    z_channels=16,
    z_factor=1,
    num_groups=1,
    legacy_mode=False,
    spatial_compression=8,
    temporal_compression=8,
    formulation=ContinuousFormulation.AE.name,
    encoder=Encoder3DType.FACTORIZED.name,
    decoder=Decoder3DType.FACTORIZED.name,
    name="CV",
)

continuous_video_8x8x8_720p = dict(continuous_video)
continuous_video_8x8x8_720p["temporal_compression"] = 8
continuous_video_8x8x8_720p["spatial_compression"] = 8

continuous_video_4x8x8_360p = dict(continuous_video)
continuous_video_4x8x8_360p["temporal_compression"] = 4
continuous_video_4x8x8_360p["spatial_compression"] = 8
continuous_video_4x8x8_360p["patch_size"] = 2


discrete_video = dict(
    attn_resolutions=[32],
    channels=128,
    channels_mult=[2, 4, 4],
    dropout=0.0,
    in_channels=3,
    num_res_blocks=2,
    out_channels=3,
    resolution=1024,
    patch_size=4,
    patch_method="haar",
    z_channels=16,
    z_factor=1,
    num_groups=1,
    legacy_mode=False,
    spatial_compression=16,
    temporal_compression=8,
    quantizer=DiscreteQuantizer.FSQ.name,
    embedding_dim=6,
    levels=[8, 8, 8, 5, 5, 5],
    encoder=Encoder3DType.FACTORIZED.name,
    decoder=Decoder3DType.FACTORIZED.name,
    name="DV",
)

discrete_video_8x16x16_720p = dict(discrete_video)
discrete_video_8x16x16_720p["temporal_compression"] = 8
discrete_video_8x16x16_720p["spatial_compression"] = 16

discrete_video_4x8x8_360p = dict(discrete_video)
discrete_video_4x8x8_360p["z_channels"] = 256
discrete_video_4x8x8_360p["temporal_compression"] = 4
discrete_video_4x8x8_360p["spatial_compression"] = 8
discrete_video_4x8x8_360p["patch_size"] = 2

# TODO: Check, this might be redundant
config = dict(
    hidden_size=768, # smaller for DEBUG, 4096 for full
    intermediate_size=768, # smaller for DEBUG, 11008 for full
    num_encoder_layers=4, # smaller for DEBUG, 16 for full
    num_decoder_layers=4, # smaller for DEBUG, 16 for full
    num_attention_heads=16, # smaller for DEBUG, 32 for full
    max_sequence_length=4096, # Not sure about this
    theta=10000.0,
    rms_norm_eps=1e-5,
    initializer_range=0.02
)

ours_discrete_video_4x8x8_256p_88 = dict(
    name="OURS",
    attn_resolutions=[32],
    channels=128,
    channels_mult=[2, 4, 4],
    dropout=0.0,
    in_channels=3,
    num_res_blocks=2,
    out_channels=3,
    resolution=1024,
    patch_size=2,
    patch_method="haar",
    # The encoder output channels just before quantization is changed to 256
    # from 16 (old versions). It aligns with the DI that uses 256 channels,
    # making initialization from image tokenizers easier.
    z_channels=256,
    z_factor=1,
    num_groups=1,
    # Most of the CV and DV tokenizers trained before September 1, 2024,
    # used temporal upsampling that was not perfectly mirrored with the
    # # encoder's temporal downsampling. Moving forward, new CV/DV tokenizers
    # will use legacy_mode=False, meaning they will adopt mirrored upsampling.
    legacy_mode=False,
    spatial_compression=8,
    temporal_compression=4,
    quantizer=DiscreteQuantizer.FSQ.name,
    embedding_dim=6,
    levels=[8, 8, 8, 5, 5, 5],
    encoder=Encoder3DType.FACTORIZED.name,
    decoder=Decoder3DType.FACTORIZED.name,
    rate_strategy='elbo',
    use_vit=True,
    freeze_original=False,
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
        method="mse"
    ),
)