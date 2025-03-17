from collections import OrderedDict

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
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.enable_sampling = enable_sampling
        self.drop_proj = drop_proj

        self.pipeline = nn.Sequential(
            OrderedDict(
                [
                    ("expansion", nn.Linear(in_channels, hidden_channels)),
                    ("activation", nn.ReLU(inplace=True)),
                    ("contraction", nn.Linear(hidden_channels, in_channels)),
                ]
            )
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.pipeline(tensor)
        tensor = nnf.dropout(
            tensor,
            p=self.drop_proj,
            training=self.training or self.enable_sampling,
        )
        return tensor
