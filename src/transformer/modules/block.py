from typing import Optional, Sequence, Type, Union

import torch
import torch.nn as nn
import torchvision.ops as tvo

from .utils import (
    make_tuple_nd,
    shift_nd,
    unshift_nd,
    batch_nd,
    unbatch_nd,
    generate_shift_nd_mask,
)
from .attention import WindowedAttention
from .feedforward import FeedForwardNetwork
from .positionalencoding import (
    RelativePositionalEncoder,
    BiasEncoder,
)


class WindowedAttentionBlockNd(nn.Module):
    def __init__(
        self,
        ndim: int,
        in_channels: int,
        dimensions: Union[int, Sequence[int]],
        kernel_size: Union[int, Sequence[int]],
        embedding_dim: int = 256,
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
        max_distance: Optional[Union[int, Sequence[int]]] = None,
    ) -> None:
        super().__init__()

        if ndim <= 0:
            raise ValueError("ndim must be positive.")

        self.ndim = ndim

        self.in_channels = in_channels
        self.out_channels = in_channels

        self.dimensions = make_tuple_nd(dimensions, ndim)
        self.kernel_size = make_tuple_nd(kernel_size, ndim)
        self.max_distance = make_tuple_nd(max_distance, ndim)

        self.shifted = shifted
        self.shift_size = [k // 2 for k in self.kernel_size]
        self.enable_sampling = enable_sampling

        self.attention = WindowedAttention(
            in_channels=in_channels,
            embedding_dim=embedding_dim,
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
        self.norm_attn = nn.LayerNorm(in_channels)

        self.mlp = FeedForwardNetwork(
            in_channels=in_channels,
            hidden_channels=in_channels * mlp_ratio,
            drop_proj=drop_proj,
            enable_sampling=enable_sampling,
            act_type=act_type,
        )
        self.norm_mlp = nn.LayerNorm(in_channels)
        self.drop_path = tvo.StochasticDepth(p=drop_path)

        self.register_buffer(
            "mask",
            generate_shift_nd_mask(
                ndim, kernel_size, dimensions, shift_size=self.shift_size
            ),
        )

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
                "or only q resulting in q, k, v to be equal."
            )

        # Volume to partition sequence
        if self.shifted:
            query = shift_nd(query, self.ndim, self.shift_size)
            key = shift_nd(key, self.ndim, self.shift_size)
            value = shift_nd(value, self.ndim, self.shift_size)

        query = batch_nd(query, self.ndim, self.kernel_size)
        key = batch_nd(key, self.ndim, self.kernel_size)
        value = batch_nd(value, self.ndim, self.kernel_size)

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
        out = unbatch_nd(out, self.ndim, self.kernel_size, self.dimensions)
        if self.shifted:
            out = unshift_nd(out, self.ndim, self.shift_size)

        return out
