from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as nnf
import einops

from .positionalencoding import RelativePositionalEncoder


class WindowAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        projection_dim: int = 256,
        heads: int = 8,
        qkv_bias: bool = True,
        drop_attn: float = 0.1,
        drop_proj: float = 0.1,
        enable_sampling: bool = False,
        rpe: RelativePositionalEncoder = None,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim // heads
        self.heads = heads
        self.qkv_bias = qkv_bias

        self.drop_attn = drop_attn
        self.drop_proj = drop_proj
        self.enable_sampling = enable_sampling

        self.proj_q = nn.Linear(embedding_dim, self.projection_dim * heads, qkv_bias)
        self.proj_k = nn.Linear(embedding_dim, self.projection_dim * heads, qkv_bias)
        self.proj_v = nn.Linear(embedding_dim, self.projection_dim * heads, False)
        self.proj_w = nn.Linear(self.projection_dim * heads, embedding_dim)

        self.scale = 1.0 / self.projection_dim**0.5
        self.rpe = rpe

        self.attention: Optional[torch.Tensor] = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        bw = query.size(1)

        query = einops.rearrange(query, "b bw ... -> (b bw) ...")
        key = einops.rearrange(key, "b bw ... -> (b bw) ...")
        value = einops.rearrange(value, "b bw ... -> (b bw) ...")

        # 1. Local embedding projection
        proj_query: torch.Tensor = einops.rearrange(
            self.proj_q(query), "b l (h e) -> b h l e", h=self.heads
        )
        proj_key: torch.Tensor = einops.rearrange(
            self.proj_k(key), "b l (h e) -> b h l e", h=self.heads
        )
        proj_value: torch.Tensor = einops.rearrange(
            self.proj_v(value), "b l (h e) -> b h l e", h=self.heads
        )

        # 2. Context computation
        context: torch.Tensor = (
            einops.einsum(proj_query, proj_key, "b h n c, b h m c -> b h n m")
            * self.scale
        )

        # 3. Relative positional encoding
        if self.rpe is not None:
            context = self.rpe(context)

        # 4. Masking
        if mask is not None:
            # b h n m -> b bw h n m
            context = einops.rearrange(context, "(b bw) ... -> b bw ...", bw=bw)
            context = context.masked_fill_(
                mask, -1000.0
            )  # No more than 1000 due to 16f support
            context = einops.rearrange(context, "b bw ... -> (b bw) ...", bw=bw)

        # 5. Compute coefficients
        attention = nnf.softmax(context, dim=-1)
        self.attention = einops.rearrange(attention, "(b bw) ... -> b bw ...", bw=bw)
        attention = nnf.dropout(
            attention,
            p=self.drop_attn,
            training=self.training or self.enable_sampling,
        )

        # 6. Compute new representation
        out = einops.einsum(attention, proj_value, "b h n m, b h m c -> b h n c")

        # 7. Project back to original dim
        out = einops.rearrange(out, "b h n c -> b n (h c)")
        out = self.proj_w(out)
        out = nnf.dropout(
            out,
            p=self.drop_proj,
            training=self.training or self.enable_sampling,
        )

        out = einops.rearrange(out, "(b bw) ... -> b bw ...", bw=bw)

        return out
