from typing import Optional
import torch
import torch.nn as nn
import einops

from ..modules.utils import (
    Input2d,
    Input3d,
    Input4d,
    make_tuple_2d,
    make_tuple_3d,
    make_tuple_4d,
)


class Batch2d(nn.Module):
    def __init__(self, kernel_size: Input2d, volume_size: Optional[Input2d] = None) -> None:
        super().__init__()

        kernel_size = make_tuple_2d(kernel_size)
        self.kernel_size = kernel_size

        self.volume_size = None
        self.window_size = None
        if volume_size is not None:
            volume_size = make_tuple_2d(volume_size)
            self.volume_size = volume_size
            self.window_size = [vs // ks for vs, ks in zip(volume_size, kernel_size)]

    def forward(self, tensor: torch.Tensor, reversed: bool = False) -> torch.Tensor:
        if not reversed:
            tensor = einops.rearrange(
                tensor,
                "b c (h hk) (w wk) -> b (h w) (hk wk) c",
                hk=self.kernel_size[0],
                wk=self.kernel_size[1],
            )
        elif self.volume_size is not None:
            tensor = einops.rearrange(
                tensor,
                "b (h w) (hk wk) c -> b c (h hk) (w wk)",
                h=self.window_size[0],
                w=self.window_size[1],
                hk=self.kernel_size[0],
                wk=self.kernel_size[1],
            )
        else:
            raise RuntimeError("Did not compute reverse. Volume required.")
        return tensor


class Batch3d(nn.Module):
    def __init__(self, kernel_size: Input3d, volume_size: Optional[Input3d] = None) -> None:
        super().__init__()

        kernel_size = make_tuple_3d(kernel_size)
        self.kernel_size = kernel_size

        self.volume_size = None
        self.window_size = None
        if volume_size is not None:
            volume_size = make_tuple_3d(volume_size)
            self.volume_size = volume_size
            self.window_size = [vs // ks for vs, ks in zip(volume_size, kernel_size)]

    def forward(self, tensor: torch.Tensor, reversed: bool = False) -> torch.Tensor:
        if not reversed:
            tensor = einops.rearrange(
                tensor,
                "b c (d dk) (h hk) (w wk) -> b (d h w) (dk hk wk) c",
                dk=self.kernel_size[0],
                hk=self.kernel_size[1],
                wk=self.kernel_size[2],
            )
        elif self.volume_size is not None:
            tensor = einops.rearrange(
                tensor,
                "b (d h w) (dk hk wk) c -> b c (d dk) (h hk) (w wk)",
                d=self.window_size[0],
                h=self.window_size[1],
                w=self.window_size[2],
                dk=self.kernel_size[0],
                hk=self.kernel_size[1],
                wk=self.kernel_size[2],
            )
        else:
            raise RuntimeError("Did not compute reverse. Volume required.")

        return tensor


class Batch4d(nn.Module):
    def __init__(
        self, kernel_size: Input4d, volume_size: Optional[Input4d] = None
    ) -> None:
        super().__init__()

        kernel_size = make_tuple_4d(kernel_size)
        self.kernel_size = kernel_size

        self.volume_size = None
        self.window_size = None
        if volume_size is not None:
            volume_size = make_tuple_4d(volume_size)
            self.volume_size = volume_size
            self.window_size = [vs // ks for vs, ks in zip(volume_size, kernel_size)]

    def forward(self, tensor: torch.Tensor, reversed: bool = False) -> torch.Tensor:
        if not reversed:
            tensor = einops.rearrange(
                tensor,
                "b c (n nk) (d dk) (h hk) (w wk) -> b (n d h w) (nk dk hk wk) c",
                nk=self.kernel_size[0],
                dk=self.kernel_size[1],
                hk=self.kernel_size[2],
                wk=self.kernel_size[3],
            )
        elif self.volume_size is not None:
            tensor = einops.rearrange(
                tensor,
                "b (n d h w) (nk dk hk wk) c -> b c (n nk) (d dk) (h hk) (w wk)",
                n=self.window_size[0],
                d=self.window_size[1],
                h=self.window_size[2],
                w=self.window_size[3],
                nk=self.kernel_size[0],
                dk=self.kernel_size[1],
                hk=self.kernel_size[2],
                wk=self.kernel_size[3],
            )
        else:
            raise RuntimeError("Did not compute reverse. Volume required.")

        return tensor
