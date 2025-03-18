import torch
import torch.nn as nn

from transformer import SwinBlock3D
from transformer.modules import Downsample

DEPTH_LOCKED = 1


class ViT(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        reduction_size=2,
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

        kernel_size = (DEPTH_LOCKED, patch_size, patch_size)
        reduct_size = (DEPTH_LOCKED, reduction_size, reduction_size)

        mut_embedding_dim = in_channels
        mut_volume_size = (DEPTH_LOCKED, img_size, img_size)

        self.embedding = Downsample(
            in_channels=in_channels,
            out_channels=mut_embedding_dim,
            volume_size=mut_volume_size,
            kernel_size=reduct_size,
            bias=True,
            type="conv",
        )

        # Update input parameter
        mut_volume_size = self.embedding.reduced_output(mut_volume_size)
        mut_embedding_dim = self.embedding.out_channels

        self.steps = nn.ModuleList()
        for i in range(len(depths)):
            print(f"mut_embedding_dim={mut_embedding_dim}")
            print(f"projection_dim={projection_dim}")
            print(f"mlp_ratio={mlp_ratio}")

            for _ in range(depths[i]):
                # Create block
                block = SwinBlock3D(
                    volume_size=mut_volume_size,
                    kernel_size=kernel_size,
                    embedding_dim=mut_embedding_dim,
                    projection_dim=projection_dim,
                    heads=heads[i],
                    mlp_ratio=mlp_ratio,
                    drop_proj=drop_proj,
                    drop_attn=drop_attn,
                    drop_path=drop_path,
                    shifted=(i % 2 == 1),
                    enable_sampling=False,
                )
                self.steps.append(block)

            # Create downsampler
            if i < len(depths) - 1:
                downsampler = Downsample(
                    in_channels=mut_embedding_dim,
                    out_channels=mut_embedding_dim * 2,
                    volume_size=mut_volume_size,
                    kernel_size=reduct_size,
                    bias=True,
                    type="concat",
                )
                self.steps.append(downsampler)

                # Update input parameter
                mut_volume_size = downsampler.reduced_output(mut_volume_size)
                mut_embedding_dim = downsampler.out_channels

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.norm = nn.LayerNorm(mut_embedding_dim)
        self.head = nn.Linear(mut_embedding_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv3d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(2)  # [B, C, 1, H, W]
        x = self.embedding(x)  # [B, embed_dim, 1, H/patch_size, W/patch_size]
        assert not torch.isnan(x).any()

        for _, stage in enumerate(self.steps):
            if isinstance(stage, SwinBlock3D):
                # For SwinBlock3D, use the same tensor for Q, K, V
                x = stage(x, x, x)
                assert not torch.isnan(x).any()
            else:
                # For Downsample
                x = stage(x)
                assert not torch.isnan(x).any()

        # Permute to [B, 1, H, W, C] for LayerNorm
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)  # Back to [B, C, 1, H, W]
        assert not torch.isnan(x).any()

        x = self.avgpool(x)  # [B, C, 1, 1, 1]
        assert not torch.isnan(x).any()
        x = torch.flatten(x, 1)  # [B, C]
        x = self.head(x)  # [B, num_classes]
        assert not torch.isnan(x).any()

        return x
