# -----------------------------------------------------------------------------
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA proprietary
# projects, please contact the Deep Imagination Research Team at
# dir@exchange.nvidia.com.
# -----------------------------------------------------------------------------

"""The network definition for discrete video tokenizer with VQ, LFQ, FSQ or ResidualFSQ. """
from collections import OrderedDict, namedtuple

import torch
from loguru import logger as logging
from torch import nn
from einops import rearrange

from cosmos_predict1.tokenizer.modules import Decoder3DType, DiscreteQuantizer, Encoder3DType
from cosmos_predict1.tokenizer.modules.layers3d import CausalConv3d
from cosmos_predict1.tokenizer.modules.layers3d_vit import TransformerBlock
from cosmos_predict1.tokenizer.modules.quantizers import InvQuantizerJit


NetworkEval = namedtuple("NetworkEval", ["reconstructions", "quant_loss", "latent", "allocated_ratios"])


class OursDiscreteVideoTokenizer(nn.Module):
    def __init__(self, z_channels: int, z_factor: int, embedding_dim: int, **kwargs) -> None:
        super().__init__()
        self.name = kwargs.get("name", "OursDiscreteVideoTokenizer")
        self.embedding_dim = embedding_dim

        encoder_name = kwargs.get("encoder", Encoder3DType.BASE.name)
        self.encoder = Encoder3DType[encoder_name].value(z_channels=z_factor * z_channels, **kwargs)

        decoder_name = kwargs.get("decoder", Decoder3DType.BASE.name)
        self.decoder = Decoder3DType[decoder_name].value(z_channels=z_channels, **kwargs)

        self.quant_conv = CausalConv3d(z_factor * z_channels, embedding_dim, kernel_size=1, padding=0)
        self.post_quant_conv = CausalConv3d(embedding_dim, z_channels, kernel_size=1, padding=0)

        quantizer_name = kwargs.get("quantizer", DiscreteQuantizer.RESFSQ.name)
        if quantizer_name == DiscreteQuantizer.VQ.name:
            assert "num_embeddings" in kwargs, f"`num_embeddings` must be provided for {quantizer_name}."
            kwargs.update(dict(embedding_dim=embedding_dim))
        elif quantizer_name == DiscreteQuantizer.LFQ.name:
            assert "codebook_size" in kwargs, f"`codebook_size` must be provided for {quantizer_name}."
            assert "codebook_dim" in kwargs, f"`codebook_dim` must be provided for {quantizer_name}."
        elif quantizer_name == DiscreteQuantizer.FSQ.name:
            assert "levels" in kwargs, f"`levels` must be provided for {quantizer_name}."
        elif quantizer_name == DiscreteQuantizer.RESFSQ.name:
            assert "levels" in kwargs, f"`levels` must be provided for {quantizer_name}."
            assert "num_quantizers" in kwargs, f"`num_quantizers` must be provided for {quantizer_name}."

        self.spatial_compression = kwargs.get("spatial_compression", 16)
        self.temporal_compression = kwargs.get("temporal_compression", 16)

        self.quantizer = DiscreteQuantizer[quantizer_name].value(**kwargs)
        logging.info(f"{self.name} based on {quantizer_name}-VAE, with {kwargs}.")

        num_parameters = sum(param.numel() for param in self.parameters())
        logging.info(f"model={self.name}, num_parameters={num_parameters:,}")
        logging.info(f"z_channels={z_channels}, embedding_dim={self.embedding_dim}.")

        self.freeze_dict = {}
        # ViT Encoder & ViT Decoder & Adaptive Tokenization
        self.rate_strategy = kwargs.get("rate_strategy", "static")
        self.use_vit = kwargs.get("use_vit", False)
        self.vit_config = kwargs.get("vit_config", None)
        # Convert vit_config dictionary to an object with attribute access
        if self.vit_config is not None and isinstance(self.vit_config, dict):
            from types import SimpleNamespace
            self.vit_config = SimpleNamespace(**self.vit_config)
        if self.use_vit:
            self.freeze_original = kwargs.get("freeze_original", True)
            if self.freeze_original:
                logging.info("Freezing original model parameters.")            
                self.freeze_dict = {k for k, v in self.named_parameters()}
                for name, parameter in self.named_parameters():
                    parameter.requires_grad = False
            else:
                logging.info("Not freezing original model parameters.")

            assert self.vit_config is not None, "vit_config must be provided if use_vit is True"
            if self.vit_config.hidden_size != z_channels:
                self.vit_conv = nn.Linear(z_channels, self.vit_config.hidden_size, bias=False)
                nn.init.normal_(self.vit_conv.weight, mean=0.0, std=self.vit_config.initializer_range)
                self.post_vit_conv = nn.Linear(self.vit_config.hidden_size, z_channels, bias=False)
                nn.init.normal_(self.post_vit_conv.weight, mean=0.0, std=self.vit_config.initializer_range)

            else:
                self.vit_conv = nn.Identity()
                self.post_vit_conv = nn.Identity()
            num_encoder_layers = self.vit_config.num_encoder_layers
            self.vit_encoder = nn.ModuleList([TransformerBlock(self.vit_config, layer_loc=i/num_encoder_layers, type='encoder') for i in range(num_encoder_layers)])
            num_decoder_layers = self.vit_config.num_decoder_layers
            self.vit_decoder = nn.ModuleList([TransformerBlock(self.vit_config, layer_loc=i/num_decoder_layers, type='decoder') for i in range(num_decoder_layers)])

            self.temporal_embed = nn.Parameter(torch.empty(self.vit_config.max_num_video_frames, self.vit_config.hidden_size))
            nn.init.normal_(self.temporal_embed, mean=0.0, std=self.vit_config.initializer_range)

            self.concat_decode_2d = self.vit_config.concat_decode_2d
            if self.concat_decode_2d:
                self.pos_emb_1d = nn.Parameter(torch.randn(self.vit_config.hidden_size) * self.vit_config.initializer_range)
                self.latent_1d_token = nn.Parameter(torch.randn(self.vit_config.max_sequence_length_1d, self.vit_config.hidden_size) * self.vit_config.initializer_range)
                self.pos_emb_2d = nn.Parameter(torch.randn(self.vit_config.hidden_size) * self.vit_config.initializer_range)
                self.init_2d_token = nn.Parameter(torch.randn(self.vit_config.max_sequence_length, self.vit_config.max_sequence_length, self.vit_config.hidden_size) * self.vit_config.initializer_range)

            self.special_attn = self.vit_config.special_attn
            self.method = self.vit_config.method
            if self.special_attn:
                self.full_latent_tokens = nn.Parameter(torch.empty(self.vit_config.max_sequence_length_1d * self.vit_config.max_num_video_frames, self.vit_config.hidden_size))
                nn.init.normal_(self.full_latent_tokens, mean=0.0, std=self.vit_config.initializer_range)
                
        self.init_adaptive_token()
        num_parameters = sum(param.numel() for param in self.parameters())
        logging.info(f"model={self.name}, num_parameters={num_parameters:,}")

    def requires_grad_(self, requires_grad=True):
        for name, parameter in self.named_parameters():
            if name not in self.freeze_dict:
                parameter.requires_grad = requires_grad
        return self

    def to(self, *args, **kwargs):
        setattr(self.quantizer, "dtype", kwargs.get("dtype", torch.bfloat16))
        return super(OursDiscreteVideoTokenizer, self).to(*args, **kwargs)

    def encoder_jit(self):
        return nn.Sequential(
            OrderedDict(
                [
                    ("encoder", self.encoder), 
                    ("quant_conv", self.quant_conv), 
                    ("quantizer", self.quantizer)
                ]
            )
        )

    def decoder_jit(self):
        return nn.Sequential(
            OrderedDict(
                [
                    ("inv_quant", InvQuantizerJit(self.quantizer)),
                    ("post_quant_conv", self.post_quant_conv),
                    ("decoder", self.decoder),
                ]
            )
        )

    def last_decoder_layer(self):
        return self.decoder.conv_out
    
    def init_adaptive_token(self, save_length: int = 128, init_value: float = 0.0):
        """
        Initialize the adaptive token with a specific value.
        Args:
            save_length (int): The length of the adaptive token.
            init_value (float): The initial value for the adaptive token.
        """
        self.register_buffer('save_loss', torch.full((save_length, ), init_value))
        self.register_buffer('ema_loss', torch.full((1,), init_value))

        logging.info(f"Adaptive token initialized with shape {self.save_loss.shape} and value {init_value}.")

    def push_signal(self, loss: torch.Tensor, beta = 0.95):
        """
        Push loss value into self.save_loss buffer. Push to the first and remove the last.
        """
        self.ema_loss = self.ema_loss * beta + (1 - beta) * loss.mean().item()
        self.save_loss = torch.roll(self.save_loss, 1, dims=0)
        self.save_loss[0] = loss.mean().item()

    def get_allocated_ratios(self, z, use_adaptive=True, chunk_loss=None, manual_base_rate: None|torch.Tensor=None, mask_seq: bool = False, overwrite_strategy: None|str = None, rescale: bool = False):
        """
        Get the allocated ratios for each block.
        Returns:
            torch.Tensor: The allocated ratios for each block.
        """
        token_shape = z.shape
        batch_size, d, num_blocks, h, w = token_shape
        if not use_adaptive:
            return torch.ones((batch_size, num_blocks))
        strategy = overwrite_strategy if overwrite_strategy is not None else self.rate_strategy

        # base_rate has size (B,)
        base_rate = torch.tensor([1.0, 0.75, 0.5, 0.25])[torch.randint(0, 4, (batch_size,))].to(z.device)
        if manual_base_rate is not None:
            base_rate = torch.ones((batch_size,), device=z.device) * manual_base_rate

        if rescale:
            assert chunk_loss is not None, "chunk_loss must be provided for rescaling"
            base_rate = self.rescale_and_push(chunk_loss.mean(dim=(-1,-2,-3)), 
                                              base_rate)

        if mask_seq:
            return base_rate[:, None]

        if 'static' in overwrite_strategy:
            allocation_ratios = torch.ones((batch_size, num_blocks,), device=z.device) * base_rate[:, None]
        elif 'uniform' in overwrite_strategy:
            rate = torch.rand(batch_size, num_blocks).clip(0.0625).to(z.device)
            rate = rate / rate.mean(dim=-1)
            allocation_ratios = (rate * base_rate[:, None])
        elif 'elbo' in strategy:
            assert chunk_loss is not None, "chunk_loss must be provided for ELBO rate strategy"
            block_loss = chunk_loss.mean(dim=(-1,-2))   # (B, T)
            base_rate = base_rate.to(block_loss.device, block_loss.dtype)
            weight = block_loss / block_loss.mean(dim=-1, keepdim=True) - 1
            max_ratio = weight.max(dim=-1, keepdim=True)[0] + 1e-5
            scale_ratio = min(max_ratio, 1 / base_rate[:, None] - 1) / max_ratio

            allocation_ratios = (1 + weight * scale_ratio) * base_rate[:, None]

        return allocation_ratios.clip(0.0625, 1.0)
    
    def mask_tokens(self, z, mask_rate: torch.Tensor, mask_method: str='order', chunk_loss=None):
        """
        Mask the tokens based on the mask rate and method.
        Args:
            z (torch.Tensor): The input tensor.
            mask_rate (torch.Tensor): The mask rate tensor.
            mask_method (str): The mask method. Can be 'order' or 'mse'.
        Returns:
            torch.Tensor: The masked tensor.
        """
        batch_size, d, _, h, w = z.shape
        mask_rate = mask_rate.reshape(batch_size, 1, -1).to(z.device)
        t = mask_rate.shape[-1]
        z = z.reshape(batch_size, d, t, -1)
        N = z.shape[-1]

        if mask_method == 'mse':
            assert chunk_loss is not None, "chunk_loss must be provided for mse method"
            assert t == 1, "mask_seq should be 1 for mse method"
            mse_loss = chunk_loss.reshape(batch_size, 1, t, -1) # chunk_loss: (B, D, 1, T*H*W), the masking is operating on the whole clip
            mse_bound = torch.quantile(mse_loss.float(), (1 - mask_rate).item(), dim=-1, keepdim=True)
            adaptive_mask = torch.where(mse_loss > mse_bound, 1.0, 0.0).to(torch.bool).detach()
        elif 'order' in mask_method:
            indices = torch.arange(N, device=mask_rate.device)
            indices = indices[None, None, None].expand(*z.shape) # z.shape: [B, D, T, H*W], indices.shape: [1, 1, 1, H*W]
            mask_rate = mask_rate.unsqueeze(-1) # [B, 1, T] -> [B, 1, T, 1]
            adaptive_mask = torch.where(indices > mask_rate * N, 0.0, 1.0).to(torch.bool).detach()

            if mask_method.startswith("order_"):
                spliter = int(mask_method.split("_")[1])
                adaptive_mask = adaptive_mask.reshape(batch_size, d, t, spliter, -1).permute(0, 1, 2, 4, 3)
                adaptive_mask = adaptive_mask.reshape(batch_size, d, t, -1)
        else:
            raise ValueError
        
        z = torch.where(adaptive_mask, z, torch.zeros_like(z))
        return z.reshape(batch_size, d, -1, h, w), adaptive_mask
    
    def adaptive_tokenize(self, z, use_adaptive=True, chunk_loss=None, manual_rate_idx: int = None, mask_seq: bool = False, method: str = 'order'):
        if method == 'mse':
            mask_seq = True
        batch_size, d, num_blocks, h, w = z.shape
        max_tokens_per_block = h * w
        if mask_seq:
            max_tokens_per_block = num_blocks * max_tokens_per_block
            num_blocks = 1
        manual_rate = [1.0, 0.75, 0.5, 0.25, 0.125][manual_rate_idx] if manual_rate_idx is not None else None
        if not use_adaptive:
            return z, torch.ones(batch_size, num_blocks, device=z.device, dtype=z.dtype)
        if self.rate_strategy == 'uniform':
            # Uniform random allocation
            allocation_ratios = torch.rand(batch_size, num_blocks, device=z.device, dtype=z.dtype).clip(0.0625)
            if manual_rate is not None:
                allocation_ratios = torch.ones_like(allocation_ratios) * manual_rate
        elif self.rate_strategy == 'unibin':
            raise ValueError("The 'unibin' rate strategy is not implemented yet.")

        elif self.rate_strategy == 'elbo':
            assert chunk_loss is not None, "chunk_loss must be provided for ELBO rate strategy"
            # ELBO rate allocation
            bins = torch.tensor([1.0, 0.75, 0.5, 0.25], 
                                          device=z.device, dtype=z.dtype)
            indices = torch.randint(0, len(bins), 
                                   (batch_size,), 
                                   device=z.device)
            allocation_ratios = bins[indices][..., None]
            if manual_rate is not None:
                allocation_ratios = torch.ones_like(allocation_ratios) * manual_rate
            block_loss = chunk_loss.mean(dim=(-1,-2))
            weight = block_loss / block_loss.mean(dim=-1, keepdim=True) - 1
            max_ratio = weight.max(dim=-1, keepdim=True)[0] + 1e-5
            scale_ratio = min(max_ratio, 1 / allocation_ratios - 1) / max_ratio
            
            allocation_ratios = allocation_ratios * (1 + weight * scale_ratio)
            
        elif self.rate_strategy == 'static':
            allocation_ratios = torch.ones(batch_size, num_blocks, device=z.device, dtype=z.dtype)
            if manual_rate is not None:
                allocation_ratios = torch.ones_like(allocation_ratios) * manual_rate
        elif 'elboema' in self.rate_strategy:
            assert mask_seq and chunk_loss is not None
            bins = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.125], device=z.device)
            if '4' in self.rate_strategy:
                bins = bins[:-1]    # remove 0.125 for the sake
            indices = torch.randint(0, len(bins), (batch_size,), device=z.device)
            allocation_ratios = bins[indices][..., None]
            
            loss = chunk_loss.mean(dim=(-1,-2,-3))
            self.push_signal(loss)
            allocation_ratios = allocation_ratios * (loss / self.ema_loss)
            allocation_ratios = allocation_ratios.clip(0.0625, 1.0)
            if 'safe' in self.rate_strategy:
                allocation_ratios = torch.where(indices == 0, 1.0, allocation_ratios)
            if torch.rand(1).item() < 0.01:
                logging.info(f"avg loss: {self.save_loss.mean():2f}, ema loss: {self.ema_loss.item():2f}, loss:{loss.mean().item():2f}, alloc_rate: {allocation_ratios.mean():2f}.")

            if manual_rate is not None:
                allocation_ratios = torch.ones_like(allocation_ratios) * manual_rate

        else:
            raise ValueError(f"Unknown rate strategy: {self.rate_strategy}")
        
        if 'order' in method:
            tokens_per_block = (allocation_ratios * max_tokens_per_block).int()
            # Clamp tokens_per_block to max_tokens_per_block
            # Create indices tensor for each position in max_tokens_per_block
            indices = torch.arange(max_tokens_per_block, device=tokens_per_block.device)
            indices = indices.unsqueeze(0).unsqueeze(0).expand(batch_size, num_blocks, -1)
            tokens_per_block = tokens_per_block.unsqueeze(-1)
            # Create mask: [1,1,1,0,0,0,0,0,0,0]
            adaptive_mask = torch.where(indices > tokens_per_block, 
                            torch.zeros_like(indices, dtype=torch.float), 
                            torch.ones_like(indices, dtype=torch.float))
            adaptive_mask = adaptive_mask.reshape(batch_size, 1, num_blocks, -1).to(torch.bool).detach()
            if method.startswith("order_"):
                spliter = int(method.split("_")[1])
                adaptive_mask = adaptive_mask.reshape(batch_size, 1, num_blocks, spliter, -1).permute(0, 1, 2, 4, 3)
                adaptive_mask = adaptive_mask.reshape(batch_size, 1, num_blocks, -1)
        elif method == 'mse':
            # For mse method, allocation_ratios may have shape [B, T] from elbo strategy
            # We reduce to [B] by taking mean across blocks
            mse_loss = chunk_loss.reshape(batch_size, -1).float()
            n = mse_loss.shape[-1]
            avg_alloc = allocation_ratios.mean(dim=-1) if allocation_ratios.dim() > 1 else allocation_ratios.flatten()
            # Vectorized quantile: sort and gather at quantile index
            sorted_loss, _ = torch.sort(mse_loss, dim=-1)
            q = (1 - avg_alloc).clamp(0, 1)  # [B]
            idx = (q * (n - 1)).long().clamp(0, n - 1)  # [B]
            mse_bound = sorted_loss.gather(dim=-1, index=idx.unsqueeze(-1))  # [B, 1]
            adaptive_mask = (mse_loss > mse_bound).reshape(batch_size, 1, num_blocks, -1).detach()
        
        z = z.reshape(batch_size, d, num_blocks, -1) * adaptive_mask
        return z.reshape(batch_size, d, -1, h, w), allocation_ratios
    
    def create_vit_mask(self, block_causal_mask, is_decoder=True):
        mask_1d = torch.concat(
            [block_causal_mask, torch.ones_like(block_causal_mask)],
            dim = 1
        )
        mask_2d = torch.concat(
            [block_causal_mask, block_causal_mask],
            dim = 1
        )
        vit_mask = torch.concat(
            [mask_1d, mask_2d] if is_decoder else [mask_2d, mask_1d],
            dim = 0
        )
        return vit_mask.to(torch.bool)
    
    def create_block_causal_mask(self, t, h, w):
        tokens_per_block = h * w  # Tokens per block
        block_mask = torch.triu(torch.ones((t, t), dtype=torch.bool), diagonal=1)
        block_causal_mask = block_mask.repeat_interleave(tokens_per_block, dim=0).repeat_interleave(tokens_per_block, dim=1)
        return block_causal_mask.to(torch.bool)
    
    def vit_encode(self, z):
        z = self.vit_conv(z.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        B, D, t, h, w = z.shape
        N = t*h*w
        attn_mask = self.create_block_causal_mask(t, h, w).to(z.device)
        # Add temporal embedding and positional embedding to 1D latent tokens
        temporal_embed = self.temporal_embed[:t].reshape(1, t, 1, -1)
        # Add temporal embedding and positional embedding to 2D tokens
        z_2d = z.permute(0, 2, 3, 4, 1).reshape(B, t, h*w, D)
        z_2d = z_2d + temporal_embed # + self.pos_emb_2d
        z = z_2d.reshape(B, t*h*w, -1)
        # Add temporal embedding and positional embedding to latent 1D tokens
        if self.concat_decode_2d:
            if not self.special_attn:  # do not consider causal 1d + full 2d
                z_1d = self.latent_1d_token[None, None, :h*w].repeat(B, t, 1, 1) 
                z_1d = z_1d + temporal_embed + self.pos_emb_1d
                attn_mask = self.create_vit_mask(attn_mask, is_decoder=False)
            else:
                z_1d = self.full_latent_tokens[None, :N].repeat(B, 1, 1)
                z_1d = z_1d + self.pos_emb_1d
                attn_mask = None

            z = torch.cat([z_1d.reshape(B, -1, z_1d.shape[-1]), z], dim=1)


        # # Concatenate and pass through ViT encoder
        for blk in self.vit_encoder:
            z = blk(
                z,
                attention_mask=attn_mask,
                position_ids=(t,h,w),
            )
        
        if self.concat_decode_2d:
            z = z[:, :N, :]
        z = self.post_vit_conv(z)
        return z.permute(0, 2, 1).reshape(B, -1, t, h, w)
        
    
    def vit_decode(self, z):
        z = self.vit_conv(z.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        B, D, t, h, w = z.shape
        attn_mask = self.create_block_causal_mask(t, h, w).to(z.device)
        temporal_embed = self.temporal_embed[:t].reshape(1, t, 1, -1)
        # Add temporal embedding and positional embedding to 1D latent tokens
        z_1d = z.permute(0, 2, 3, 4, 1).reshape(B, t, h*w, D)
        if not self.special_attn:
            z_1d = z_1d + temporal_embed # + self.pos_emb_1d
        z = z_1d.reshape(B, t*h*w, -1)
        # Add temporal embedding and positional embedding to 2D tokens
        if self.concat_decode_2d:
            z_2d = self.init_2d_token[:h, :w].reshape(h*w, -1)
            z_2d = z_2d[None, None].repeat(B, t, 1, 1)
            z_2d = z_2d + temporal_embed + self.pos_emb_2d
            z = torch.cat([z, z_2d.reshape(B, t*h*w, -1)], dim=1)

            attn_mask = self.create_vit_mask(attn_mask, is_decoder=True)

        if self.special_attn:
            attn_mask = None

        # # Concatenate and pass through ViT decoder
        for blk in self.vit_decoder:
            z = blk(
                z,
                attention_mask=attn_mask,
                position_ids=(t,h,w),
            )

        if self.concat_decode_2d:
            z = z[:, -t*h*w:, :]
        z = self.post_vit_conv(z)
        return z.permute(0, 2, 1).reshape(B, -1, t, h, w)
        

    def encode(self, x):
        z = self.encoder(x) # [B, D, t, h, w], continuous
        B, D, t, h, w = z.shape
        if self.use_vit:
            z = self.vit_encode(z)
        z = self.quant_conv(z) # [B, n_quan, t, h, w], continuous
        return self.quantizer(z)

    def decode(self, quant):
        z = self.post_quant_conv(quant) # input is [B, n_quan, t, h, w], continuous
        B, D, t, h, w = z.shape
        if self.use_vit:
            z = self.vit_decode(z) # The self.vit_mask is processed in the encode function
        z = self.decoder(z) # input is [B, D, t, h, w], continuous
        return z

    def decode_code(self, code_b):
        quant_b = self.quantizer.indices_to_codes(code_b)
        quant_b = self.post_quant_conv(quant_b)
        z = self.decoder(quant_b)
        return z

    def forward(self, input, n_tries: int = 0):
        quant_info, quant_codes, quant_loss = self.encode(input)
        chunk_loss = None
        if 'elbo' in self.rate_strategy:
            with torch.no_grad():
                recon_full = self.decode(quant_codes)
                full_loss = (input - recon_full).abs().mean(dim=1)
                chunk_loss = full_loss[:, 1:].reshape(full_loss.shape[0], -1, self.temporal_compression, *full_loss.shape[2:]).mean(dim=2)
                chunk_loss = torch.concat([full_loss[:, :1], chunk_loss], dim=1)
                chunk_loss = rearrange(chunk_loss, "b t (h u) (w v) -> b t h u w v", u=self.spatial_compression, v=self.spatial_compression).mean(dim=(-1,-3))
                # chunk loss has shape [B, T, H, W]
        quant_codes, allocated_ratios = self.adaptive_tokenize(quant_codes, use_adaptive=True, chunk_loss=chunk_loss, manual_rate_idx=None if self.training else n_tries, mask_seq=self.special_attn, method=self.method)

        reconstructions = self.decode(quant_codes)
        if self.training:
            return dict(reconstructions=reconstructions, quant_loss=quant_loss, latent=quant_codes, allocated_ratios=allocated_ratios.float())
        return NetworkEval(reconstructions=reconstructions, quant_loss=quant_loss, latent=quant_codes, allocated_ratios=allocated_ratios.float())