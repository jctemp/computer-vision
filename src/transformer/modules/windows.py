import torch
import torch.nn as nn
from einops import rearrange

from ..utils import Dimensions3d


class WindowPartition3d(nn.Module):
    def __init__(self, volume_size: Dimensions3d, kernel_size: Dimensions3d) -> None:
        super().__init__()
        self.volume_size = volume_size
        self.kernel_size = kernel_size
        self.window_size = Dimensions3d(
            volume_size.depth // kernel_size.depth,
            volume_size.height // kernel_size.height,
            volume_size.width // kernel_size.width,
        )
        assert volume_size.divisible_by(kernel_size), (
            f"Not all dimensions are divisible by kernel, received: {volume_size} and {kernel_size}"
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        assert len(tensor.size()) == 5, (
            f"Require a volume with channels and batch (BCDHW), received: {tensor.shape}"
        )

        kernel_depth = self.kernel_size.depth
        kernel_height = self.kernel_size.height
        kernel_width = self.kernel_size.width

        tensor = rearrange(
            tensor,
            "b c (d kd) (h kh) (w kw) -> b (d h w) (kd kh kw) c",
            kd=kernel_depth,
            kh=kernel_height,
            kw=kernel_width,
        )

        return tensor


class WindowReverse3d(nn.Module):
    def __init__(self, volume_size: Dimensions3d, kernel_size: Dimensions3d) -> None:
        super().__init__()
        self.volume_size = volume_size
        self.kernel_size = kernel_size
        self.window_size = Dimensions3d(
            volume_size.depth // kernel_size.depth,
            volume_size.height // kernel_size.height,
            volume_size.width // kernel_size.width,
        )
        assert volume_size.divisible_by(kernel_size), (
            f"Not all dimensions are divisible by kernel, received: {volume_size} and {kernel_size}"
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        assert len(tensor.size()) == 4, (
            f"Require a sequence (BWLE), received: {tensor.shape}"
        )

        depth = self.window_size.depth
        height = self.window_size.height
        width = self.window_size.width

        kernel_depth = self.kernel_size.depth
        kernel_height = self.kernel_size.height
        kernel_width = self.kernel_size.width

        tensor = rearrange(
            tensor,
            "b (d h w) (kd kh kw) c -> b c (d kd) (h kh) (w kw)",
            d=depth,
            h=height,
            w=width,
            kd=kernel_depth,
            kh=kernel_height,
            kw=kernel_width,
        )

        return tensor


class WindowShift3d(nn.Module):
    def __init__(self, volume_size: Dimensions3d, kernel_size: Dimensions3d) -> None:
        super().__init__()
        self.volume_size = volume_size
        self.kernel_size = kernel_size
        self.shift_size = Dimensions3d(
            -self.kernel_size.depth // 2,
            -self.kernel_size.height // 2,
            -self.kernel_size.width // 2,
        )
        self.shift_size_rev = Dimensions3d(
            self.kernel_size.depth // 2,
            self.kernel_size.height // 2,
            self.kernel_size.width // 2,
        )
        self.mask = self._generate_mask(volume_size, kernel_size)

    def _generate_mask(
        self, volume_size: Dimensions3d, kernel_size: Dimensions3d, device=None
    ) -> torch.BoolTensor:
        vol2seq = WindowPartition3d(volume_size, kernel_size).to(device)

        # 1. Generate index map
        index_map = torch.zeros(
            (
                self.volume_size.depth,
                self.volume_size.height,
                self.volume_size.width,
            ),
            device=device,
        )

        slices_depth = [
            (0, -self.kernel_size.depth),
            (-self.kernel_size.depth, -self.shift_size.depth),
            (-self.shift_size.depth, None),
        ]
        slices_height = [
            (0, -self.kernel_size.height),
            (-self.kernel_size.height, -self.shift_size.height),
            (-self.shift_size.height, None),
        ]
        slices_width = [
            (0, -self.kernel_size.width),
            (-self.kernel_size.width, -self.shift_size.width),
            (-self.shift_size.width, None),
        ]

        count = 0
        for d_start, d_stop in slices_depth:
            for h_start, h_stop in slices_height:
                for w_start, w_stop in slices_width:
                    index_map[d_start:d_stop, h_start:h_stop, w_start:w_stop] = count
                    count += 1

        index_map_batched: torch.Tensor = (
            vol2seq(index_map.unsqueeze(0).unsqueeze(0)).squeeze(-1).squeeze(0)
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
            tensor = torch.roll(
                tensor, shifts=self.shift_size.to_tuple(), dims=(2, 3, 4)
            )
        else:
            tensor = torch.roll(
                tensor, shifts=self.shift_size_rev.to_tuple(), dims=(2, 3, 4)
            )

        return tensor
