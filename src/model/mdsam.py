import torch
from torch import nn
from .mask_decoder import MaskDecoder
from .image_encoder import ImageEncoderViT
from .transformer import TwoWayTransformer
from .common import LayerNorm2d, PositionEmbeddingRandom

from typing import Any, Optional, Tuple, Type
from functools import partial
from reprlib import recursive_repr
import numpy as np
from torch.nn import functional as F
from .block import MLFusion, DetailEnhancement


class MDSAM(nn.Module):
    def __init__(self, img_size = 512, norm = nn.BatchNorm2d, act = nn.ReLU):
        super().__init__()

        self.pe_layer = PositionEmbeddingRandom(256 // 2)

        self.image_embedding_size = [img_size // 16, img_size // 16]
        self.img_size = img_size

        self.image_encoder = ImageEncoderViT(depth=12,
            embed_dim=768,
            img_size=img_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=12,
            patch_size=16,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[2, 5, 8, 11],
            window_size=14,
            out_chans=256)

        self.mask_decoder = MaskDecoder(
            transformer=TwoWayTransformer(
                depth = 2,
                embedding_dim = 256,
                mlp_dim = 2048,
                num_heads = 8
            ),
            transformer_dim=256,
            norm = norm,
            act = act
        )
        self.deep_feautre_conv = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = False),
            nn.Conv2d(256, 64, 3, padding = 1, bias = False),
            norm(64),
            act(),
            nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = False),
            nn.Conv2d(64, 32, 3, padding = 1, bias = False),
            norm(32),
            act(),
        )
        self.deep_out_conv = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding = 1, bias = False),
            norm(16),
            act(),
            nn.Conv2d(16, 1, 1)
        )

        self.fusion_block = MLFusion(norm = norm, act = act)

        self.detail_enhance = DetailEnhancement(img_dim = 32, feature_dim = 32, norm = norm, act = act)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def forward(self, img):
        #get different layer's features of the image encoder
        features_list = self.image_encoder(img)

        #get the output of the last layer of the encoder.
        deep_feature = self.deep_feautre_conv(features_list[-1].contiguous()) #256 * 32 * 32 -> 32 * 128 * 128

        img_feature = self.fusion_block(features_list)

        #extract intermediate outputs for deep supervision to prevent model overfitting on the detail enhancement module.
        img_pe = self.get_dense_pe()
        coarse_mask, feature= self.mask_decoder(img_feature, img_pe)

        coarse_mask = torch.nn.functional.interpolate(coarse_mask,[self.img_size,self.img_size], mode = 'bilinear', align_corners = False)

        mask = self.detail_enhance(img, feature, deep_feature)

        return mask, coarse_mask
