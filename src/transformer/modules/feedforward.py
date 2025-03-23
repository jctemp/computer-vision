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

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.expand.weight, std=0.02)
        if hasattr(self.expand, "bias") and self.expand.bias is not None:
            nn.init.constant_(self.expand.bias, 0)

        nn.init.trunc_normal_(self.contract.weight, std=0.02)
        if hasattr(self.contract, "bias") and self.contract.bias is not None:
            nn.init.constant_(self.contract.bias, 0)

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
