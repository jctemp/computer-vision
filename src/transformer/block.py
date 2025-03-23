from typing import Optional, Type

import torch
import torch.nn as nn

from .modules import (
    Shift2d,
    Shift3d,
    Shift4d,
    Batch2d,
    Batch3d,
    Batch4d,
    WindowAttention,
    RelativePositionalEncoder,
    BiasEncoder,
    FeedForwardNetwork,
    DropPath,
)
from .modules.utils import InputNd, make_tuple_2d, make_tuple_3d, make_tuple_4d

make_tuple = {
    2: make_tuple_2d,
    3: make_tuple_3d,
    4: make_tuple_4d,
}

Shift = {
    2: Shift2d,
    3: Shift3d,
    4: Shift4d,
}

Batch = {
    2: Batch2d,
    3: Batch3d,
    4: Batch4d,
}


class WindowAttentionBlock(nn.Module):
    def __init__(
        self,
        volume_size: InputNd,
        kernel_size: InputNd,
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
        act_type: Type[nn.Module] = nn.GELU,
        rpe_type: Type[RelativePositionalEncoder] = BiasEncoder,
        max_distance: Optional[InputNd] = None,
    ) -> None:
        super().__init__()

        dim = -1
        if isinstance(volume_size, tuple) or isinstance(volume_size, list):
            dim = len(volume_size)
        elif isinstance(kernel_size, tuple) or isinstance(kernel_size, list):
            dim = len(kernel_size)
        elif isinstance(max_distance, tuple) or isinstance(max_distance, list):
            dim = len(max_distance)
        else:
            raise ValueError(
                "Cannot infer dimension. Please pass a tuple or list to compute current Nd."
            )

        self.in_channels = embedding_dim
        self.out_channels = embedding_dim

        self.volume_size = make_tuple[dim](volume_size)
        self.kernel_size = make_tuple[dim](kernel_size)
        self.max_distance = make_tuple[dim](max_distance)

        self.shifted = shifted
        self.enable_sampling = enable_sampling

        self.partition = Batch[dim](self.kernel_size, self.volume_size)
        self.shift = Shift[dim](self.kernel_size, self.volume_size)

        self.attention = WindowAttention(
            embedding_dim=embedding_dim,
            projection_dim=projection_dim,
            heads=heads,
            qkv_bias=qkv_bias,
            drop_attn=drop_attn,
            drop_proj=drop_proj,
            enable_sampling=enable_sampling,
            rpe=rpe_type(
                self.kernel_size,
                heads,
                self.max_distance,
            ),
        )
        self.norm_attn = nn.LayerNorm(embedding_dim)

        self.mlp = FeedForwardNetwork(
            in_channels=embedding_dim,
            hidden_channels=embedding_dim * mlp_ratio,
            drop_proj=drop_proj,
            enable_sampling=enable_sampling,
            act_type=act_type,
        )
        self.norm_mlp = nn.LayerNorm(embedding_dim)
        self.drop_path = DropPath(p=drop_path, enable_sampling=enable_sampling)

        self.register_buffer("mask", self.shift.mask)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.constant_(self.norm_attn.weight, 1.0)
        nn.init.constant_(self.norm_attn.bias, 0)

        nn.init.constant_(self.norm_mlp.weight, 0.0)
        nn.init.constant_(self.norm_mlp.bias, 0.0)

        if self.attention is not None and hasattr(self.attention, "_init_weights"):
            self.attention._init_weights()

        if self.mlp is not None and hasattr(self.mlp, "_init_weights"):
            self.mlp._init_weights()

    def forward(
        self,
        query: torch.Tensor,  # b c d h w
        key: Optional[torch.Tensor] = None,  # b c d h w
        value: Optional[torch.Tensor] = None,  # b c d h w
        mask: Optional[torch.BoolTensor] = None,  # b c d h w
    ) -> torch.Tensor:
        if key is None and value is None:
            key = query
            value = query
        else:
            raise ValueError(
                "You have to set query(q), key(k) and value(v) "
                "or only q resulting in q, k, v to be equi."
            )

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
        out = self.partition(out, reversed=True)
        if self.shifted:
            out = self.shift(out, reversed=True)

        return out
