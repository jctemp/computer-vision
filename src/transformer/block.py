from typing import Optional, Tuple

import torch
import torch.nn as nn

from .modules import (
    WindowAttention3d,
    WindowPartition3d,
    WindowReverse3d,
    WindowShift3d,
    FeedForwardNetwork,
    DropPath,
)
from .utils import Dimensions3d


class SwinBlock3D(nn.Module):
    def __init__(
        self,
        volume_size: Dimensions3d | Tuple[int, int, int],
        kernel_size: Dimensions3d | Tuple[int, int, int],
        embedding_dim: int,
        projection_dim: int = 256,
        heads: int = 8,
        qkv_bias: bool = True,
        drop_attn: float = 0.1,
        drop_proj: float = 0.1,
        drop_path: float = 0.1,
        mlp_ratio: int = 3,
        shifted: bool = False,
        enable_sampling: bool = False,
    ) -> None:
        super().__init__()

        self.in_channels = embedding_dim
        self.out_channels = embedding_dim

        self.volume_size = (
            volume_size
            if isinstance(volume_size, Dimensions3d)
            else Dimensions3d.fromtuple(volume_size)
        )
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, Dimensions3d)
            else Dimensions3d.fromtuple(kernel_size)
        )
        self.shifted = shifted
        self.enable_sampling = enable_sampling

        self.partition = WindowPartition3d(
            volume_size=self.volume_size,
            kernel_size=self.kernel_size,
        )
        self.reverse = WindowReverse3d(
            volume_size=self.volume_size,
            kernel_size=self.kernel_size,
        )
        self.shift = WindowShift3d(
            volume_size=self.volume_size,
            kernel_size=self.kernel_size,
        )

        self.attention = WindowAttention3d(
            embedding_dim=embedding_dim,
            projection_dim=projection_dim,
            heads=heads,
            qkv_bias=qkv_bias,
            drop_attn=drop_attn,
            drop_proj=drop_proj,
            enable_sampling=enable_sampling,
        )
        self.norm_attn = nn.LayerNorm(embedding_dim)

        self.mlp = FeedForwardNetwork(
            in_channels=embedding_dim,
            hidden_channels=embedding_dim * mlp_ratio,
            enable_sampling=enable_sampling,
        )
        self.norm_mlp = nn.LayerNorm(embedding_dim)

        self.drop_path = DropPath(p=drop_path, enable_sampling=enable_sampling)

        self.register_buffer(
            "indices",
            Dimensions3d.compute_relative_positions(self.kernel_size),
        )
        self.register_buffer("mask", self.shift.mask)

    def forward(
        self,
        query: torch.Tensor,  # b c d h w
        key: torch.Tensor,  # b c d h w
        value: torch.Tensor,  # b c d h w
        mask: Optional[torch.BoolTensor] = None,  # b c d h w
    ) -> torch.Tensor:
        # Volume to partition sequence
        if self.shifted:
            query = self.shift(query)
            key = self.shift(key)
            value = self.shift(value)

        query = self.partition(query)
        key = self.partition(key)
        value = self.partition(value)

        masked = self.mask
        if mask is not None:
            masked += self.partition(mask)

        # Attention computation
        residual = query
        attention = self.attention(
            query=query,
            key=key,
            value=value,
            indices=self.indices,
            mask=masked,
        )
        attention = self.norm_attn(attention)
        attention = self.drop_path(attention) + residual

        # MLP processing
        residual = attention
        out = self.mlp(attention)
        out = self.norm_mlp(out)
        out = self.drop_path(out) + residual

        # Sequence to volume
        out = self.reverse(out)
        if self.shifted:
            out = self.shift(out, reversed=True)

        return out
