import torch
import torch.nn as nn
import torch.nn.functional as nnf
import einops

from ..modules.utils import (
    TupleNd,
    Input2d,
    Input3d,
    Input4d,
    make_tuple_2d,
    make_tuple_3d,
    make_tuple_4d,
)


class Merge(nn.Module):
    def reduced_output(self, volume_size: TupleNd) -> TupleNd:
        return tuple([v // k for v, k in zip(volume_size, self.kernel_size)])


class Merge2d(Merge):
    def __init__(
        self,
        in_channels: int,
        kernel_size: Input2d,
        drop_proj: float = 0.05,
        enable_sampling: bool = False,
    ) -> None:
        super().__init__()

        kernel_size = make_tuple_2d(kernel_size)

        total = in_channels * kernel_size[0] * kernel_size[1]
        half = total // 2

        self.in_channels = in_channels
        self.out_channels = half
        self.kernel_size = kernel_size
        self.drop_proj = drop_proj
        self.enable_sampling = enable_sampling

        self.proj = nn.Linear(total, half, bias=True)
        self.norm = nn.LayerNorm(half)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        if hasattr(self.proj, "bias") and self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0)

        if hasattr(self.norm, "weight") and self.norm.weight is not None:
            nn.init.constant_(self.norm.weight, 1.0)
        if hasattr(self.norm, "bias") and self.norm.bias is not None:
            nn.init.constant_(self.norm.bias, 0)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = einops.rearrange(
            tensor,
            "b c (h hk) (w wk) -> b h w (hk wk c)",
            hk=self.kernel_size[0],
            wk=self.kernel_size[1],
        )

        tensor = self.proj(tensor)
        tensor = self.norm(tensor)
        tensor = nnf.dropout(
            tensor,
            p=self.drop_proj,
            training=self.training or self.enable_sampling,
        )

        tensor = einops.rearrange(tensor, "b h w c -> b c h w")

        return tensor


class Merge3d(Merge):
    def __init__(
        self,
        in_channels: int,
        kernel_size: Input3d,
        drop_proj: float = 0.05,
        enable_sampling: bool = False,
    ) -> None:
        super().__init__()

        kernel_size = make_tuple_3d(kernel_size)

        total = in_channels * kernel_size[0] * kernel_size[1] * kernel_size[2]
        half = total // 2

        self.in_channels = in_channels
        self.out_channels = half
        self.kernel_size = kernel_size
        self.drop_proj = drop_proj
        self.enable_sampling = enable_sampling

        self.proj = nn.Linear(total, half, bias=True)
        self.norm = nn.LayerNorm(half)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        if hasattr(self.proj, "bias") and self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0)

        if hasattr(self.norm, "weight") and self.norm.weight is not None:
            nn.init.constant_(self.norm.weight, 1.0)
        if hasattr(self.norm, "bias") and self.norm.bias is not None:
            nn.init.constant_(self.norm.bias, 0)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = einops.rearrange(
            tensor,
            "b c (d dk) (h hk) (w wk) -> b d h w (dk hk wk c)",
            dk=self.kernel_size[0],
            hk=self.kernel_size[1],
            wk=self.kernel_size[2],
        )

        tensor = self.proj(tensor)
        tensor = self.norm(tensor)
        tensor = nnf.dropout(
            tensor,
            p=self.drop_proj,
            training=self.training or self.enable_sampling,
        )

        tensor = einops.rearrange(tensor, "b d h w c -> b c d h w")

        return tensor


class Merge4d(Merge):
    def __init__(
        self,
        in_channels: int,
        kernel_size: Input4d,
        drop_proj: float = 0.05,
        enable_sampling: bool = False,
    ) -> None:
        super().__init__()

        kernel_size = make_tuple_4d(kernel_size)

        total = (
            in_channels
            * kernel_size[0]
            * kernel_size[1]
            * kernel_size[2]
            * kernel_size[3]
        )
        half = total // 2

        self.in_channels = in_channels
        self.out_channels = half
        self.kernel_size = kernel_size
        self.drop_proj = drop_proj
        self.enable_sampling = enable_sampling

        self.proj = nn.Linear(total, half, bias=True)
        self.norm = nn.LayerNorm(half)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        if hasattr(self.proj, "bias") and self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0)

        if hasattr(self.norm, "weight") and self.norm.weight is not None:
            nn.init.constant_(self.norm.weight, 1.0)
        if hasattr(self.norm, "bias") and self.norm.bias is not None:
            nn.init.constant_(self.norm.bias, 0)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = einops.rearrange(
            tensor,
            "b c (n nk) (d dk) (h hk) (w wk) -> b n d h w (nk dk hk wk c)",
            nk=self.kernel_size[0],
            dk=self.kernel_size[1],
            hk=self.kernel_size[2],
            wk=self.kernel_size[3],
        )

        tensor = self.proj(tensor)
        tensor = self.norm(tensor)
        tensor = nnf.dropout(
            tensor,
            p=self.drop_proj,
            training=self.training or self.enable_sampling,
        )

        tensor = einops.rearrange(tensor, "b n d h w c -> b c n d h w")

        return tensor
