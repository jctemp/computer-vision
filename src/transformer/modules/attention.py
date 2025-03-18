from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as nnf

from einops import rearrange, einsum


class WindowAttention3d(nn.Module):
    """Window-based multi-head self-attention for 3D volumes with relative position bias.

    This implements the attention mechanism from Swin Transformer V2 with:
    - Cosine similarity attention with learned temperature
    - Log-spaced continuous position bias
    - Support for shifted window attention masking
    """

    def __init__(
        self,
        embedding_dim: int,
        projection_dim: int = 256,
        heads: int = 8,
        qkv_bias: bool = True,
        drop_attn: float = 0.1,
        drop_proj: float = 0.1,
        enable_sampling: bool = False,
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

        self.scale = nn.Parameter(torch.ones(1))
        self.cpb_mlp = nn.Sequential(
            nn.Linear(3, 512), nn.ReLU(inplace=True), nn.Linear(512, heads)
        )
        self.attention: Optional[torch.Tensor] = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        indices: torch.IntTensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        bw = query.size(1)

        query = rearrange(query, "b bw ... -> (b bw) ...")
        key = rearrange(key, "b bw ... -> (b bw) ...")
        value = rearrange(value, "b bw ... -> (b bw) ...")

        # 1. Local embedding projection
        proj_query: torch.Tensor = rearrange(
            self.proj_q(query), "b l (h e) -> b h l e", h=self.heads
        )
        proj_key: torch.Tensor = rearrange(
            self.proj_k(key), "b l (h e) -> b h l e", h=self.heads
        )
        proj_value: torch.Tensor = rearrange(
            self.proj_v(value), "b l (h e) -> b h l e", h=self.heads
        )

        # 2. Cosine-similarity
        norm_query = nnf.normalize(proj_query, dim=-1)
        norm_key = nnf.normalize(proj_key, dim=-1)
        context: torch.Tensor = einsum(
            norm_query, norm_key, "b h n c, b h m c -> b h n m"
        ) / self.scale.clamp(min=0.01)

        # 3. Log continous positional bias
        log_indices = (
            # magic number 8 -- dunno why
            torch.sign(indices)
            * torch.log2(1 + indices.abs())
            / torch.log2(torch.tensor(8))
        )
        bias = 16 * torch.sigmoid(self.cpb_mlp(log_indices))
        bias = rearrange(bias, "l s h -> 1 h l s")
        context += bias

        # 4. masking
        if mask is not None:
            # b h n m -> b bw h n m
            context = rearrange(context, "(b bw) ... -> b bw ...", bw=bw)
            context = context.masked_fill_(mask, -1e5)
            context = rearrange(context, "b bw ... -> (b bw) ...", bw=bw)

        # 5. Compute coefficients
        attention = nnf.softmax(context, dim=-1)
        self.attention = attention
        attention = nnf.dropout(
            attention,
            p=self.drop_attn,
            training=self.training or self.enable_sampling,
        )

        # 6. Compute new representation
        out = einsum(attention, proj_value, "b h n m, b h m c -> b h n c")

        # 7. Project back to original dim
        out = rearrange(out, "b h n c -> b n (h c)")
        out = self.proj_w(out)
        out = nnf.dropout(
            out,
            p=self.drop_proj,
            training=self.training or self.enable_sampling,
        )

        out = rearrange(out, "(b bw) ... -> b bw ...", bw=bw)

        return out
