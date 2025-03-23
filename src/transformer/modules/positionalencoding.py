from typing import Optional
import functools

import torch
import torch.nn as nn
import einops


from ..modules.utils import (
    InputNd,
    compute_relative_positions_2d,
    compute_relative_positions_3d,
    compute_relative_positions_4d,
)


class RelativePositionalEncoder(nn.Module):
    def __init__(
        self, kernel_size: InputNd, heads: int, max_distance: Optional[InputNd] = None
    ) -> None:
        super().__init__()

        self.kernel_size = kernel_size
        self.heads = heads
        self.max_distance = (
            [k - 1 for k in kernel_size]
            if max_distance is None
            else [min(d, k) for d, k in zip(max_distance, kernel_size)]
        )

    def get_indices(self) -> torch.Tensor:
        indices = None
        if len(self.kernel_size) == 2:
            indices = compute_relative_positions_2d(
                self.kernel_size, self.max_distance, False
            )
        elif len(self.kernel_size) == 3:
            indices = compute_relative_positions_3d(
                self.kernel_size, self.max_distance, False
            )
        elif len(self.kernel_size) == 4:
            indices = compute_relative_positions_4d(
                self.kernel_size, self.max_distance, False
            )
        else:
            raise ValueError("kernel_size is not of dim: 2, 3, 4")
        return indices

    def get_embeddings(self, _: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("get_embeddings() is not implemented")

    def _post_embedding(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor += einops.rearrange(self.get_embeddings(tensor), "l s h -> () h l s")
        return self._post_embedding(tensor)


class BiasEncoder(RelativePositionalEncoder):
    def __init__(
        self, kernel_size: InputNd, heads: int, max_distance: Optional[InputNd] = None
    ) -> None:
        super().__init__(kernel_size, heads, max_distance)

        self.positional_bias = nn.Embedding(
            sum(2 * d + 1 for d in self.max_distance), self.heads
        )
        self.register_buffer("indices", self.get_indices())

        self._init_weights()

    def get_embeddings(self, _: torch.Tensor) -> torch.Tensor:
        return self.positional_bias(self.indices.type(torch.int64)).sum(-2)

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.positional_bias.weight, std=0.02)


class ContinuousEncoder(RelativePositionalEncoder):
    def __init__(
        self, kernel_size: InputNd, heads: int, max_distance: Optional[InputNd] = None
    ) -> None:
        super().__init__(kernel_size, heads, max_distance)

        hidden_dim = functools.reduce(lambda x, y: x * y, self.kernel_size)
        self.cpb_mlp = nn.Sequential(
            nn.Linear(len(self.kernel_size), hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.heads),
        )

        indices = self.get_indices()
        log_indices = (
            # use log_8
            torch.sign(indices)
            * torch.log2(1 + indices.abs())
            / torch.log2(torch.tensor(8))
        )
        self.register_buffer("indices", log_indices)

        self._init_weights()

    def get_embeddings(self, _: torch.Tensor) -> torch.Tensor:
        return 16 * torch.sigmoid(self.cpb_mlp(self.indices))

    def _init_weights(self) -> None:
        for m in self.cpb_mlp:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
