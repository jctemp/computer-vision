from dataclasses import dataclass
from typing import Tuple, Self

from einops import repeat
import torch


@dataclass
class Dimensions3d:
    depth: int
    height: int
    width: int

    def fromvalue(self, value: int) -> Self:
        return Dimensions3d(value, value, value)

    def num_elements(self) -> int:
        return self.depth * self.height * self.width

    def divisible_by(self, dims: Self) -> bool:
        return (
            self.depth % dims.depth == 0
            and self.height % dims.height == 0
            and self.width % dims.width == 0
        )

    def to_tuple(self) -> Tuple[int, int, int]:
        return (self.depth, self.height, self.width)

    def compute_absolute_coordinates(self, device=None) -> torch.Tensor:
        indices_depth = repeat(
            torch.arange(self.depth, device=device),
            "c -> () c h w",
            h=self.height,
            w=self.width,
        )
        indices_height = repeat(
            torch.arange(self.height, device=device),
            "c -> () d c w",
            d=self.depth,
            w=self.width,
        )
        indices_width = repeat(
            torch.arange(self.width, device=device),
            "c -> () d h c",
            d=self.depth,
            h=self.height,
        )
        coords = torch.concat([indices_depth, indices_height, indices_width], dim=0)

        return coords

    def compute_relative_positions(
        self, norm: bool = True, device=None
    ) -> torch.Tensor:
        indices_depth = repeat(
            torch.arange(self.depth, dtype=torch.float, device=device),
            "c -> c h w",
            h=self.height,
            w=self.width,
        ).flatten()
        indices_height = repeat(
            torch.arange(self.height, dtype=torch.float, device=device),
            "c -> d c w",
            d=self.depth,
            w=self.width,
        ).flatten()
        indices_width = repeat(
            torch.arange(self.width, dtype=torch.float, device=device),
            "c -> d h c",
            d=self.depth,
            h=self.height,
        ).flatten()

        relative_depth = indices_depth.unsqueeze(0) - indices_depth.unsqueeze(1)
        relative_height = indices_height.unsqueeze(0) - indices_height.unsqueeze(1)
        relative_width = indices_width.unsqueeze(0) - indices_width.unsqueeze(1)

        if norm:
            relative_depth /= self.depth - 1
            relative_height /= self.height - 1
            relative_width /= self.width - 1

        coords = torch.stack([relative_depth, relative_height, relative_width], dim=-1)

        return coords
