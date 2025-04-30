from typing import Optional, Type, Union, Sequence
import math

import torch
import torch.nn as nn
import torch.nn.functional as nnf
import einops

from .utils import make_tuple_nd


class DownsampleNd(nn.Module):
    def __init__(
        self,
        ndim: int,
        in_channels: int,
        kernel_size: Union[int, Sequence[int]],
        out_channels: Optional[int] = None,
        drop_proj: float = 0.05,
        enable_sampling: bool = False,
    ) -> None:
        super().__init__()

        if ndim <= 0:
            raise ValueError("ndim must be positive.")

        self.ndim = ndim
        kernel_size = make_tuple_nd(kernel_size, ndim)

        kernel_product = math.prod(kernel_size)
        total_in_features = in_channels * kernel_product

        out_channels = total_in_features // 2 if out_channels is None else out_channels

        if out_channels == 0:
            raise ValueError("out_channels must be at least 1.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.drop_proj = drop_proj
        self.enable_sampling = enable_sampling

        self.proj = nn.Linear(total_in_features, out_channels, bias=True)
        self.norm = nn.LayerNorm(out_channels)

        self._init_weights()

        # Input:  b c (d1 k1) (d2 k2) ... (dn kn)
        # Output: b d1 d2 ... dn (k1 k2 ... kn c)
        spatial_dims_in = " ".join([f"(d{i} k{i})" for i in range(ndim)])
        spatial_dims_out = " ".join([f"d{i}" for i in range(ndim)])
        kernel_channels_out = " ".join([f"k{i}" for i in range(ndim)]) + " c"
        self.rearrange_in_pattern = (
            f"b c {spatial_dims_in} -> b {spatial_dims_out} ({kernel_channels_out})"
        )

        # Input:  b d1 d2 ... dn c_out
        # Output: b c_out d1 d2 ... dn
        self.rearrange_out_pattern = (
            f"b {spatial_dims_out} c_out -> b c_out {spatial_dims_out}"
        )

        self.rearrange_params = {f"k{i}": ks for i, ks in enumerate(self.kernel_size)}

    def _init_weights(self) -> None:
        # Using standard initialization from your original code
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        if hasattr(self.proj, "bias") and self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0)

        if hasattr(self.norm, "weight") and self.norm.weight is not None:
            nn.init.constant_(self.norm.weight, 1.0)
        if hasattr(self.norm, "bias") and self.norm.bias is not None:
            nn.init.constant_(self.norm.bias, 0)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        expected_dims = self.ndim + 2  # (batch, channel, spatial_dims...)
        if tensor.ndim != expected_dims:
            raise ValueError(
                f"Input tensor expected to have {expected_dims} dimensions "
                f"(B, C{' Dim'.join([','] * (self.ndim))} Dim), but got {tensor.ndim}"
            )

        for i in range(self.ndim):
            if tensor.shape[2 + i] % self.kernel_size[i] != 0:
                raise ValueError(
                    f"Input tensor dimension {i + 2} (size {tensor.shape[2 + i]}) "
                    f"is not divisible by corresponding kernel_size {self.kernel_size[i]}"
                )

        # Rearrange: Fold patches into channel dimension
        tensor = einops.rearrange(
            tensor,
            self.rearrange_in_pattern,
            **self.rearrange_params,
        )

        # Project, Normalize, Dropout
        tensor = self.proj(tensor)
        tensor = self.norm(tensor)
        tensor = nnf.dropout(
            tensor,
            p=self.drop_proj,
            training=self.training or self.enable_sampling,
        )

        # Rearrange: Move channels back to dimension 1
        tensor = einops.rearrange(tensor, self.rearrange_out_pattern)

        return tensor


class LayerNormNd(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.constant_(self.norm.weight, 1.0)
        nn.init.constant_(self.norm.bias, 0)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = einops.rearrange(tensor, "b c ... -> b ... c")
        tensor = self.norm(tensor)
        tensor = einops.rearrange(tensor, "b ... c -> b c ...")
        return tensor


class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        drop_proj: float = 0.1,
        enable_sampling: bool = False,
        act_type: Type[torch.nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.enable_sampling = enable_sampling
        self.drop_proj = drop_proj

        self.expand = nn.Linear(in_channels, hidden_channels)
        self.activate = act_type()
        self.contract = nn.Linear(hidden_channels, in_channels)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.expand.weight, std=0.02)
        if hasattr(self.expand, "bias") and self.expand.bias is not None:
            nn.init.constant_(self.expand.bias, 0)

        nn.init.trunc_normal_(self.contract.weight, std=0.02)
        if hasattr(self.contract, "bias") and self.contract.bias is not None:
            nn.init.constant_(self.contract.bias, 0)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.expand(tensor)
        tensor = self.activate(tensor)
        tensor = self.contract(tensor)

        tensor = nnf.dropout(
            tensor,
            p=self.drop_proj,
            training=self.training or self.enable_sampling,
        )

        return tensor
