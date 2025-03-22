from typing import Type
import torch
import torch.nn as nn
import torch.nn.functional as nnf


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
