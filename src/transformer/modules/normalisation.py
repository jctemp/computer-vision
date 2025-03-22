import torch
import torch.nn as nn
from einops import rearrange

class LayerNormNd(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = rearrange(tensor, "b c ... -> b ... c")
        tensor = self.norm(tensor)
        tensor = rearrange(tensor, "b ... c -> b c ...")
        return tensor
