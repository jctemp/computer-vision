import torch
import torch.nn as nn


class DropPath(nn.Module):
    def __init__(
        self,
        p: float,
        inplace: bool = False,
        enable_sampling: bool = False,
    ) -> None:
        super().__init__()
        self.p = p
        self.inplace = inplace
        self.enable_sampling = enable_sampling

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.training or self.enable_sampling:
            tshape = (tensor.size(0),) + (1,) * (tensor.ndim - 1)
            mask: torch.Tensor = tensor.new_empty(tshape).bernoulli_(1.0 - self.p)
            mask /= 1.0 - self.p

            if self.inplace:
                tensor *= mask
            else:
                tensor = tensor * mask
        return tensor
