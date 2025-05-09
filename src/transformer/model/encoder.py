from typing import List, Optional, Sequence, Type, Union

import torch
import torch.nn as nn
from transformer.modules import (
    WindowedAttentionBlockNd,
    DownsampleNd,
    BiasEncoder,
    RelativePositionalEncoder,
    LayerNormNd,
)


class EncoderNd(nn.Module):
    def __init__(
        self,
        ndim: int = 3,
        # Embedding layer
        in_channels: int = 3,
        initial_embedding_dim: int = 256,
        initial_reduction_kernel_size: Union[int, Sequence[int]] = 2,
        # Stage specific parameters
        stage_depths: List[int] = [2, 2, 4, 2],
        stage_heads: List[int] = [4, 4, 8, 4],
        stage_window_sizes: List[Union[int, Sequence[int]]] = [8, 8, 8, 8],
        stage_reduction_kernel_sizes: List[Union[int, Sequence[int]]] = [2, 2, 2],
        # Stage general parameters
        mlp_ratio: int = 4,
        qkv_bias: bool = True,
        drop_attn_rate: float = 0.0,
        drop_proj_rate: float = 0.0,
        stochastic_depth_base_rate: float = 0.1,
        enable_sampling_for_dropout: bool = False,
        act_type: Type[nn.Module] = nn.GELU,
        rpe_type: Type[RelativePositionalEncoder] = BiasEncoder,
        max_rpe_distance: Optional[Union[int, Sequence[int]]] = None,
        reduction_out_channels_multiplier: int = 2,
    ):
        super().__init__()

        if ndim <= 0:
            raise ValueError("ndim must be positive.")
        current_channels = in_channels
        self.embedding = DownsampleNd(
            ndim=ndim,
            in_channels=current_channels,
            kernel_size=initial_reduction_kernel_size,
            out_channels=initial_embedding_dim,
            drop_proj=0,
            enable_sampling=enable_sampling_for_dropout,
        )
        current_channels = self.embedding.out_channels

        num_stages = len(stage_depths)
        total_blocks = sum(stage_depths)

        sd_probs = [
            x.item()
            for x in torch.linspace(0, stochastic_depth_base_rate, total_blocks)
        ]
        block_sd_idx_counter = 0

        if not (
            len(stage_heads) == num_stages and len(stage_window_sizes) == num_stages
        ):
            raise ValueError(
                "stage_depths, stage_heads, and stage_window_sizes must have the same length."
            )
        if len(stage_reduction_kernel_sizes) != num_stages - 1 and num_stages > 1:
            raise ValueError(
                f"stage_reduction_kernel_sizes should be of length {num_stages - 1} if num_stages > 1, got {len(stage_reduction_kernel_sizes)}"
            )

        self.stages = nn.ModuleList()
        for stage_idx in range(num_stages):
            stage_specific_modules = nn.ModuleDict()

            if stage_idx > 0:
                reduction_k_size = stage_reduction_kernel_sizes[stage_idx - 1]
                if reduction_k_size is not None:
                    reduction = DownsampleNd(
                        ndim=ndim,
                        in_channels=current_channels,
                        kernel_size=reduction_k_size,
                        out_channels=current_channels
                        * reduction_out_channels_multiplier
                        if isinstance(reduction_k_size, int) and reduction_k_size > 1
                        else current_channels,
                        drop_proj=drop_proj_rate,
                        enable_sampling=enable_sampling_for_dropout,
                    )
                    stage_specific_modules.add_module("reduction", reduction)
                    current_channels = reduction.out_channels

            blocks_sequence = nn.Sequential()
            for block_idx in range(stage_depths[stage_idx]):
                block_drop_path_rate = sd_probs[block_sd_idx_counter]
                block_sd_idx_counter += 1

                block = WindowedAttentionBlockNd(
                    ndim=ndim,
                    in_channels=current_channels,
                    kernel_size=stage_window_sizes[stage_idx],
                    embedding_dim=current_channels,
                    heads=stage_heads[stage_idx],
                    qkv_bias=qkv_bias,
                    drop_attn=drop_attn_rate,
                    drop_proj=drop_proj_rate,
                    drop_path=block_drop_path_rate,
                    mlp_ratio=mlp_ratio,
                    shifted=(block_idx % 2 == 1),
                    enable_sampling=enable_sampling_for_dropout,
                    act_type=act_type,
                    rpe_type=rpe_type,
                    max_distance=max_rpe_distance,
                )
                blocks_sequence.add_module(f"b{block_idx}", block)

            stage_specific_modules.add_module("blocks", blocks_sequence)
            self.stages.append(stage_specific_modules)

        self.norm = LayerNormNd(current_channels)
        self.out_channels = current_channels
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.embedding(x)

        stage_feature_maps = []

        for stage_module_dict in self.stages:
            if "reduction" in stage_module_dict:
                x = stage_module_dict["reduction"](x)
            x = stage_module_dict["blocks"](x)
            stage_feature_maps.append(x)

        if not self.stages:
            if x is None:
                raise ValueError(
                    "Encoder has no embedding layer or stages, and x is None."
                )
            normalized_output = self.norm(x)
            return [normalized_output]

        normalized_final_output = self.norm(stage_feature_maps[-1])
        final_outputs_to_return = stage_feature_maps[:-1] + [normalized_final_output]

        return final_outputs_to_return
