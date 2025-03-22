import torch
import torch.nn as nn
from einops import rearrange

from ..modules.utils import (
    Input2d,
    Input3d,
    Input4d,
    make_tuple_2d,
    make_tuple_3d,
    make_tuple_4d,
)
from ..modules.batch import Batch2d, Batch3d, Batch4d


class Shift2d(nn.Module):
    def __init__(self, kernel_size: Input2d, volume_size: Input2d) -> None:
        super().__init__()

        kernel_size = make_tuple_2d(kernel_size)
        self.kernel_size = kernel_size

        volume_size = make_tuple_2d(volume_size)
        self.volume_size = volume_size

        self.shift_size = (
            -self.kernel_size[0] // 2,
            -self.kernel_size[1] // 2,
        )
        self.shift_size_rev = (
            self.kernel_size[0] // 2,
            self.kernel_size[1] // 2,
        )
        self.mask = self._generate_mask(kernel_size)

    def _generate_mask(self, kernel_size: Input2d, device=None) -> torch.BoolTensor:
        batch = Batch2d(kernel_size=kernel_size)

        # 1. Generate index map
        index_map = torch.zeros(
            (
                self.volume_size[0],
                self.volume_size[1],
            ),
            device=device,
        )

        slices_height = [
            (0, -self.kernel_size[0]),
            (-self.kernel_size[0], -self.shift_size[0]),
            (-self.shift_size[0], None),
        ]
        slices_width = [
            (0, -self.kernel_size[1]),
            (-self.kernel_size[1], -self.shift_size[1]),
            (-self.shift_size[1], None),
        ]

        count = 0
        for h_start, h_stop in slices_height:
            for w_start, w_stop in slices_width:
                index_map[h_start:h_stop, w_start:w_stop] = count
                count += 1

        index_map_batched: torch.Tensor = (
            batch(index_map.unsqueeze(0).unsqueeze(0)).squeeze(-1).squeeze(0)
        )

        # 2. Compare indices to find window differences
        ids = index_map_batched.unsqueeze(1) - index_map_batched.unsqueeze(2)

        # 3. Mask non-equal window sections
        ids = ids.masked_fill(ids != 0, True).masked_fill(ids == 0, False)

        # 4. Retrieve broadcastable shape for attention computation
        ids = rearrange(ids, "bw ... -> () bw () ...")

        return ids.to(torch.bool)

    def forward(self, tensor: torch.Tensor, reversed: bool = False) -> torch.Tensor:
        if not reversed:
            tensor = torch.roll(tensor, shifts=self.shift_size, dims=(2, 3))
        else:
            tensor = torch.roll(tensor, shifts=self.shift_size_rev, dims=(2, 3))

        return tensor


class Shift3d(nn.Module):
    def __init__(self, kernel_size: Input3d, volume_size: Input3d) -> None:
        super().__init__()

        kernel_size = make_tuple_3d(kernel_size)
        self.kernel_size = kernel_size

        volume_size = make_tuple_3d(volume_size)
        self.volume_size = volume_size

        self.shift_size = (
            -self.kernel_size[0] // 2,
            -self.kernel_size[1] // 2,
            -self.kernel_size[2] // 2,
        )
        self.shift_size_rev = (
            self.kernel_size[0] // 2,
            self.kernel_size[1] // 2,
            self.kernel_size[2] // 2,
        )
        self.mask = self._generate_mask(kernel_size)

    def _generate_mask(self, kernel_size: Input3d, device=None) -> torch.BoolTensor:
        batch = Batch3d(kernel_size=kernel_size)

        # 1. Generate index map
        index_map = torch.zeros(
            (
                self.volume_size[0],
                self.volume_size[1],
                self.volume_size[2],
            ),
            device=device,
        )

        slices_depth = [
            (0, -self.kernel_size[0]),
            (-self.kernel_size[0], -self.shift_size[0]),
            (-self.shift_size[0], None),
        ]
        slices_height = [
            (0, -self.kernel_size[1]),
            (-self.kernel_size[1], -self.shift_size[1]),
            (-self.shift_size[1], None),
        ]
        slices_width = [
            (0, -self.kernel_size[2]),
            (-self.kernel_size[2], -self.shift_size[2]),
            (-self.shift_size[2], None),
        ]

        count = 0
        for d_start, d_stop in slices_depth:
            for h_start, h_stop in slices_height:
                for w_start, w_stop in slices_width:
                    index_map[d_start:d_stop, h_start:h_stop, w_start:w_stop] = count
                    count += 1

        index_map_batched: torch.Tensor = (
            batch(index_map.unsqueeze(0).unsqueeze(0)).squeeze(-1).squeeze(0)
        )

        # 2. Compare indices to find window differences
        ids = index_map_batched.unsqueeze(1) - index_map_batched.unsqueeze(2)

        # 3. Mask non-equal window sections
        ids = ids.masked_fill(ids != 0, True).masked_fill(ids == 0, False)

        # 4. Retrieve broadcastable shape for attention computation
        ids = rearrange(ids, "bw ... -> () bw () ...")

        return ids.to(torch.bool)

    def forward(self, tensor: torch.Tensor, reversed: bool = False) -> torch.Tensor:
        if not reversed:
            tensor = torch.roll(tensor, shifts=self.shift_size, dims=(2, 3, 4))
        else:
            tensor = torch.roll(tensor, shifts=self.shift_size_rev, dims=(2, 3, 4))

        return tensor


class Shift4d(nn.Module):
    def __init__(self, kernel_size: Input4d, volume_size: Input4d) -> None:
        super().__init__()

        kernel_size = make_tuple_4d(kernel_size)
        self.kernel_size = kernel_size

        volume_size = make_tuple_4d(volume_size)
        self.volume_size = volume_size

        self.shift_size = (
            -self.kernel_size[0] // 2,
            -self.kernel_size[1] // 2,
            -self.kernel_size[2] // 2,
            -self.kernel_size[3] // 2,
        )
        self.shift_size_rev = (
            self.kernel_size[0] // 2,
            self.kernel_size[1] // 2,
            self.kernel_size[2] // 2,
            self.kernel_size[3] // 2,
        )
        self.mask = self._generate_mask(kernel_size)

    def _generate_mask(self, kernel_size: Input3d, device=None) -> torch.BoolTensor:
        batch = Batch4d(kernel_size=kernel_size)

        # 1. Generate index map
        index_map = torch.zeros(
            (
                self.volume_size[0],
                self.volume_size[1],
                self.volume_size[2],
                self.volume_size[3],
            ),
            device=device,
        )

        slices_n = [
            (0, -self.kernel_size[0]),
            (-self.kernel_size[0], -self.shift_size[0]),
            (-self.shift_size[0], None),
        ]
        slices_d = [
            (0, -self.kernel_size[1]),
            (-self.kernel_size[1], -self.shift_size[1]),
            (-self.shift_size[1], None),
        ]
        slices_h = [
            (0, -self.kernel_size[2]),
            (-self.kernel_size[2], -self.shift_size[2]),
            (-self.shift_size[2], None),
        ]
        slices_w = [
            (0, -self.kernel_size[3]),
            (-self.kernel_size[3], -self.shift_size[3]),
            (-self.shift_size[3], None),
        ]

        count = 0
        for n_start, n_stop in slices_n:
            for d_start, d_stop in slices_d:
                for h_start, h_stop in slices_h:
                    for w_start, w_stop in slices_w:
                        index_map[
                            n_start:n_stop,
                            d_start:d_stop,
                            h_start:h_stop,
                            w_start:w_stop,
                        ] = count
                        count += 1

        index_map_batched: torch.Tensor = (
            batch(index_map.unsqueeze(0).unsqueeze(0)).squeeze(-1).squeeze(0)
        )

        # 2. Compare indices to find window differences
        ids = index_map_batched.unsqueeze(1) - index_map_batched.unsqueeze(2)

        # 3. Mask non-equal window sections
        ids = ids.masked_fill(ids != 0, True).masked_fill(ids == 0, False)

        # 4. Retrieve broadcastable shape for attention computation
        ids = rearrange(ids, "bw ... -> () bw () ...")

        return ids.to(torch.bool)

    def forward(self, tensor: torch.Tensor, reversed: bool = False) -> torch.Tensor:
        if not reversed:
            tensor = torch.roll(tensor, shifts=self.shift_size, dims=(2, 3, 4, 5))
        else:
            tensor = torch.roll(tensor, shifts=self.shift_size_rev, dims=(2, 3, 4, 5))

        return tensor
