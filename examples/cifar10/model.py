import torch
import torch.nn as nn

from transformer import WindowAttentionBlock
from transformer.modules import Merge2d, ContinuousEncoder


class ShiftingWindowTransformer(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=(4, 4, 4, 2),
        reduction_size=(2, 2, 2, 2),
        in_channels=3,
        num_classes=10,
        embed_dim=256,
        depths=(2, 2, 4, 2),
        num_heads=(4, 4, 8, 4),
        mlp_ratio=4,
        dropout=0.1,
    ) -> None:
        super().__init__()

        projection_dim = embed_dim
        heads = num_heads
        mlp_ratio = mlp_ratio
        drop_proj = dropout
        drop_attn = dropout
        drop_path = 0.0

        kernel_size = [(p, p) for p in patch_size]
        reduct_size = [(r, r) for r in reduction_size]

        mut_embedding_dim = in_channels
        mut_volume_size = (img_size, img_size)

        self.embedding = Merge2d(
            in_channels=in_channels,
            kernel_size=reduct_size[0],
            drop_proj=0,
            enable_sampling=False,
        )

        # Update input parameter
        mut_volume_size = self.embedding.reduced_output(mut_volume_size)
        mut_embedding_dim = self.embedding.out_channels

        self.steps = nn.ModuleList()
        for i in range(len(depths)):
            for k in range(depths[i]):
                # Create block
                block = WindowAttentionBlock(
                    volume_size=mut_volume_size,
                    kernel_size=kernel_size[i],
                    in_channels=mut_embedding_dim,
                    embedding_dim=projection_dim,
                    heads=heads[i],
                    mlp_ratio=mlp_ratio,
                    drop_proj=drop_proj,
                    drop_attn=drop_attn,
                    drop_path=drop_path,
                    shifted=(k % 2 == 1),
                    enable_sampling=False,
                    rpe_type=ContinuousEncoder,
                )
                self.steps.append(block)

            # Create downsampler
            if i < len(depths) - 1:
                downsampler = Merge2d(
                    in_channels=mut_embedding_dim,
                    kernel_size=reduct_size[i + 1],
                    drop_proj=drop_proj,
                    enable_sampling=False,
                )
                self.steps.append(downsampler)

                # Update input parameter
                mut_volume_size = downsampler.reduced_output(mut_volume_size)
                mut_embedding_dim = downsampler.out_channels

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.norm = nn.LayerNorm(mut_embedding_dim)
        self.head = nn.Linear(mut_embedding_dim, num_classes)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        assert not torch.isnan(x).any()

        for _, stage in enumerate(self.steps):
            if isinstance(stage, WindowAttentionBlock):
                x = stage(x)
            else:
                x = stage(x)

        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)

        return x
