from cosmos_predict1.utils.lazy_config import LazyCall as L
from cosmos_predict1.utils.lazy_config import LazyDict
from cosmos_predict1.tokenizer.modules import (
    ContinuousFormulation,
    Decoder3DType,
    DecoderType,
    DiscreteQuantizer,
    Encoder3DType,
    EncoderType,
)

from cosmos_predict1.tokenizer.networks.ours_discrete_video import OursDiscreteVideoTokenizer


AdaptiveDiscreteVideoTokenizerConfig: LazyDict = L(OursDiscreteVideoTokenizer)(
    # Matches ours_discrete_video_4x8x8_mse_256p from configs.py (OURS4x8x8-mse-256p-88)
    # Base network settings
    name="OURS",
    attn_resolutions=[32],
    channels=128,
    channels_mult=[2, 4, 4],
    dropout=0.0,
    in_channels=3,
    num_res_blocks=2,
    out_channels=3,
    resolution=1024,
    patch_size=2,           # Changed from 4
    patch_method="haar",    # Changed from "rearrange"
    z_channels=256,
    z_factor=1,
    num_groups=1,
    legacy_mode=False,
    # Compression settings (4x8x8)
    temporal_compression=4,  # Changed from 8
    spatial_compression=8,   # Changed from 16
    # Encoder/Decoder use FACTORIZED (base), ViT is added via use_vit
    encoder=Encoder3DType.FACTORIZED.name,
    decoder=Decoder3DType.FACTORIZED.name,
    # Quantizer parameters (FSQ)
    quantizer=DiscreteQuantizer.FSQ.name,
    embedding_dim=6,
    levels=[8, 8, 8, 5, 5, 5],
    # Adaptive tokenization settings
    rate_strategy="elbo",
    use_vit=True,
    freeze_original=False,
    # ViT backbone configuration (8 layers encoder/decoder, mse method)
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
)