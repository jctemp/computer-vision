from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from einops import rearrange

from ..utils import Dimensions3d


class Concat3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Optional[Union[int, Tuple[int, int, int]]] = None,
        padding: Optional[Union[str, int, Tuple[int, int, int]]] = 0,
    ) -> None:
        super().__init__()

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if stride is None:
            self.stride = self.kernel_size
        elif isinstance(stride, int):
            self.stride = (stride, stride, stride)
        else:
            self.stride = stride

        if isinstance(padding, int):
            self.padding = (padding, padding, padding)
        else:
            self.padding = padding

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.proj = nn.Linear(
            in_channels * kernel_size[0] * kernel_size[1] * kernel_size[2],
            out_channels,
            True,
        )
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        _, _, depth, height, width = tensor.shape
        kernel_depth, kernel_height, kernel_width = self.kernel_size
        stride_depth, stride_height, stride_width = self.stride

        if self.padding == "same":
            # Calculate output dimensions with "same" padding
            out_depth = (depth + stride_depth - 1) // stride_depth
            out_height = (height + stride_height - 1) // stride_height
            out_width = (width + stride_width - 1) // stride_width

            # Calculate required padding
            pad_depth = max(0, (out_depth - 1) * stride_depth + kernel_depth - depth)
            pad_height = max(
                0, (out_height - 1) * stride_height + kernel_height - height
            )
            pad_width = max(0, (out_width - 1) * stride_width + kernel_width - width)

            pad_depth_front = pad_depth // 2
            pad_depth_back = pad_depth - pad_depth_front
            pad_height_top = pad_height // 2
            pad_height_bottom = pad_height - pad_height_top
            pad_width_left = pad_width // 2
            pad_width_right = pad_width - pad_width_left

            tensor = nnf.pad(
                tensor,
                (
                    pad_width_left,
                    pad_width_right,
                    pad_height_top,
                    pad_height_bottom,
                    pad_depth_front,
                    pad_depth_back,
                ),
            )
        else:
            pad_depth, pad_height, pad_width = self.padding

            pad_depth_front = pad_depth // 2
            pad_depth_back = pad_depth - pad_depth_front
            pad_height_top = pad_height // 2
            pad_height_bottom = pad_height - pad_height_top
            pad_width_left = pad_width // 2
            pad_width_right = pad_width - pad_width_left

            tensor = nnf.pad(
                tensor,
                (
                    pad_width_left,
                    pad_width_right,
                    pad_height_top,
                    pad_height_bottom,
                    pad_depth_front,
                    pad_depth_back,
                ),
            )

            assert (
                tensor.size(-3) % kernel_depth == 0
            ), f"Padding leads to invalid image dimensions, got: {tensor.size(-3) % kernel_depth}"
            assert (
                tensor.size(-2) % kernel_height == 0
            ), f"Padding leads to invalid image dimensions, got: {tensor.size(-2) % kernel_height}"
            assert (
                tensor.size(-1) % kernel_width == 0
            ), f"Padding leads to invalid image dimensions, got: {tensor.size(-1) % kernel_width}"

        tensor = rearrange(
            tensor,
            "... c (d kd) (h kh) (w kw) -> ... d h w (c kd kh kw)",
            kd=kernel_depth,
            kh=kernel_height,
            kw=kernel_width,
        )

        tensor = self.proj(tensor)
        tensor = self.norm(tensor)

        tensor = rearrange(tensor, "... d h w c -> ... c d h w")

        return tensor


class Downsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        volume_size: Dimensions3d,
        kernel_size: Dimensions3d,
        bias: bool = True,
        type: str = "con",
    ) -> None:
        super().__init__()

        self.volume_size = volume_size
        self.kernel_size = kernel_size
        self.bias = bias

        self.pooler = None
        if type == "max":
            self.pooler = nn.MaxPool3d(
                kernel_size=kernel_size.to_tuple(),
                stride=kernel_size.to_tuple(),
            )
        elif type == "avg":
            self.pooler = nn.AvgPool3d(
                kernel_size=kernel_size.to_tuple(),
                stride=kernel_size.to_tuple(),
            )
        elif type == "cnn":
            self.pooler = nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size.to_tuple(),
                stride=kernel_size.to_tuple(),
            )
        elif type == "con":
            self.pooler = Concat3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size.to_tuple(),
                stride=kernel_size.to_tuple(),
            )
        else:
            raise TypeError()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.pooler(tensor)
        return tensor
