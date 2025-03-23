import torch
import torch.nn as nn
from einops import rearrange


class LayerNormNd(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.constant_(self.norm.weight, 1.0)
        nn.init.constant_(self.norm.bias, 0)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = rearrange(tensor, "b c ... -> b ... c")
        tensor = self.norm(tensor)
        tensor = rearrange(tensor, "b ... c -> b c ...")
        return tensor
