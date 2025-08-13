import torch
import numpy as np
from torch import nn
from functools import partial
from torch.nn import functional as F
from typing import Any, Optional, Tuple

from .mask_decoder import MaskDecoder
from .image_encoder import ImageEncoderViT
from .transformer import TwoWayTransformer
from .common import PositionEmbeddingRandom


class DefectSAM(nn.Module):
    def __init__(
        self, 
        img_size=384):
        super().__init__()

        self.image_embedding_size = [img_size // 16, img_size // 16]

        self.image_encoder = ImageEncoderViT(
            img_size=img_size,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            use_rel_pos=True,
            global_attn_indexes=[2, 5, 8, 11],
            window_size=14,
        )

        self.mask_decoder = MaskDecoder(
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
        )

        self.pe_layer = PositionEmbeddingRandom(128)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def postprocess_masks(
        self,
        masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )

        return masks
    
    def forward(self, img):
        # image encoder
        img_embeddings = self.image_encoder(img)

        # mask decoder
        img_pe = self.get_dense_pe()
        pred = self.mask_decoder(img_embeddings, img_pe)
        pred = self.postprocess_masks(pred)

        return torch.sigmoid(pred)