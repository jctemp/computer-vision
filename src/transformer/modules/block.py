from typing import Optional, Sequence, Type, Union, Dict, Tuple

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
from .layers import FeedForwardNetwork
from .positionalencoding import (
    RelativePositionalEncoder,
    BiasEncoder,
)


class WindowedAttentionBlockNd(nn.Module):
    def __init__(
        self,
        ndim: int,
        in_channels: int,
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

        self.kernel_size = make_tuple_nd(kernel_size, ndim)
        self.max_distance = (
            make_tuple_nd(max_distance, ndim) if max_distance is not None else None
        )

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
                ndim,
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
        self.drop_path = tvo.StochasticDepth(p=drop_path, mode="row")

        self.mask: Dict[Tuple[int, ...], torch.BoolTensor] = {}

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.constant_(self.norm_attn.weight, 1.0)
        nn.init.constant_(self.norm_attn.bias, 0)

        nn.init.constant_(self.norm_mlp.weight, 1.0)
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
        mask: Optional[torch.BoolTensor] = None,  # b d h w
    ) -> torch.Tensor:
        input_spatial_dims = tuple(query.size()[2:])

        external_mask: Optional[torch.BoolTensor] = None
        cached_shift_mask: Optional[torch.BoolTensor] = None

        if key is None and value is None:
            key = query
            value = query
        else:
            raise ValueError(
                "You have to set query(q), key(k) and value(v) "
                "or only q resulting in q, k, v to be equal."
            )

        if mask is not None:
            external_mask = mask.unsqueeze(1).to(query.device)

            # 1. Shift the mask if the input is shifted
            if self.shifted:
                external_mask = shift_nd(external_mask, self.ndim, self.shift_size)

            # 2. Build patches as the windows are masked individuallys
            batched_external_mask, _ = batch_nd(
                external_mask, self.ndim, self.kernel_size
            )

            # 3. Compute mask for query and keys
            patch_product_len = batched_external_mask.size(2)
            batched_external_mask = batched_external_mask.squeeze(-1)
            # (B, num_windows, patch_prod_len, 1)
            mask_q_expanded = batched_external_mask.unsqueeze(-1).expand(
                -1, -1, -1, patch_product_len
            )
            # (B, num_windows, 1, patch_prod_len)
            mask_k_expanded = batched_external_mask.unsqueeze(-2).expand(
                -1, -1, patch_product_len, -1
            )

            # 4. If either has been mask, we cannot use it in attention computation
            # (B, num_windows, patch_prod_len, patch_prod_len)
            external_mask = mask_q_expanded | mask_k_expanded

            # 5. Have a broadcastable shape
            # (B, num_windows, 1, patch_prod_len, patch_prod_len)
            external_mask = external_mask.unsqueeze(2)

        # Volume to partition sequence
        if self.shifted:
            cached_shift_mask = self.mask.get(input_spatial_dims)
            if cached_shift_mask is None:
                cached_shift_mask = generate_shift_nd_mask(
                    self.ndim,
                    self.kernel_size,
                    input_spatial_dims,
                    shift_size=self.shift_size,
                )
                self.mask[input_spatial_dims] = cached_shift_mask

            query = shift_nd(query, self.ndim, self.shift_size)
            key = shift_nd(key, self.ndim, self.shift_size)
            value = shift_nd(value, self.ndim, self.shift_size)

        query_batched, batched_spatial_dims = batch_nd(
            query, self.ndim, self.kernel_size
        )
        key_batched, _ = batch_nd(key, self.ndim, self.kernel_size)
        value_batched, _ = batch_nd(value, self.ndim, self.kernel_size)

        attention_mask = None
        if external_mask is not None:
            attention_mask = external_mask
        if cached_shift_mask is not None:
            if attention_mask is not None:
                attention_mask = attention_mask | cached_shift_mask
            else:
                attention_mask = cached_shift_mask

        # Attention computation
        residual = query_batched
        attention = self.attention(
            query=query_batched,
            key=key_batched,
            value=value_batched,
            mask=attention_mask,
        )
        attention = self.norm_attn(attention)
        attention = self.drop_path(attention) + residual

        # MLP processing
        residual = attention
        out = self.mlp(attention)
        out = self.norm_mlp(out)
        out = self.drop_path(out) + residual

        # Sequence to volume
        out = unbatch_nd(out, self.ndim, self.kernel_size, batched_spatial_dims)
        if self.shifted:
            out = unshift_nd(out, self.ndim, self.shift_size)

        return out
