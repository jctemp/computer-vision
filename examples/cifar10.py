import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchmetrics.classification import MulticlassAccuracy
from typing import Optional, Tuple

from transformer.model import EncoderNd
from transformer.modules import (
    BiasEncoder,
    ContinuousEncoder,
)


class ImageClassifier(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        self.encoder = EncoderNd(
            ndim=2,
            in_channels=3,
            initial_embedding_dim=96,
            initial_reduction_kernel_size=4,
            stage_depths=[2, 2, 6],
            stage_heads=[3, 6, 12],
            stage_window_sizes=[
                4,
                2,
                2,
            ],
            stage_reduction_kernel_sizes=[
                2,
                2,
            ],
            mlp_ratio=4,
            qkv_bias=True,
            drop_attn_rate=0.0,
            drop_proj_rate=0.0,
            stochastic_depth_base_rate=0.1,
            act_type=nn.GELU,
            rpe_type=BiasEncoder,
            max_rpe_distance=None,
            reduction_out_channels_multiplier=2,
        )

        self.pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.fc = nn.Linear(self.encoder.out_channels, num_classes)

        self.accuracy = MulticlassAccuracy(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features_list = self.encoder(x)
        x = features_list[-1]
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.accuracy(logits, y)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.accuracy(logits, y)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_acc", acc, prog_bar=True, logger=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.accuracy(logits, y)
        self.log("test_loss", loss, logger=True)
        self.log("test_acc", acc, logger=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(
        self, data_dir: str = "./data", batch_size: int = 128, num_workers: int = 4
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # CIFAR-10 specific normalization
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        self.dims = (3, 32, 32)  # Input dimensions

    def prepare_data(self):
        # Download CIFAR-10
        torchvision.datasets.CIFAR10(self.data_dir, train=True, download=True)
        torchvision.datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            cifar_full = torchvision.datasets.CIFAR10(
                self.data_dir, train=True, transform=self.transform
            )
            # Split training into train and validation
            train_size = int(0.9 * len(cifar_full))  # 90% for training
            val_size = len(cifar_full) - train_size
            self.cifar_train, self.cifar_val = random_split(
                cifar_full, [train_size, val_size]
            )

        if stage == "test" or stage is None:
            self.cifar_test = torchvision.datasets.CIFAR10(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.cifar_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.cifar_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.cifar_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )


if __name__ == "__main__":
    pl.seed_everything(42)  # For reproducibility

    # --- Data Module ---
    cifar_dm = CIFAR10DataModule(
        batch_size=256, num_workers=os.cpu_count() // 2 if os.cpu_count() else 2
    )

    # --- Model ---
    # Ensure your EncoderNd and its dependencies are correctly defined and importable
    # The parameters for EncoderNd are crucial and depend on its implementation.
    # The ones provided in ImageClassifier are illustrative.
    model = ImageClassifier(learning_rate=1e-3, num_classes=10)

    # --- Trainer ---
    # For faster testing, you can limit epochs, use a single GPU, or CPU
    # For CPU: accelerator="cpu", devices=1
    # For GPU: accelerator="gpu", devices=1 (or more if available)

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("Using MPS (Apple Silicon GPU)")
        accelerator = "mps"
        devices = 1
    elif torch.cuda.is_available():
        print("Using CUDA GPU")
        accelerator = "gpu"
        devices = 1  # Or [0] for specific GPU, or "auto"
    else:
        print("Using CPU")
        accelerator = "cpu"
        devices = 1

    trainer = pl.Trainer(
        max_epochs=10,  # Keep low for a quick test run
        accelerator=accelerator,
        devices=devices,
        logger=pl.loggers.CSVLogger("logs/"),  # Simple CSV logger
        # callbacks=[pl.callbacks.ModelCheckpoint(monitor='val_loss')], # Optional: save checkpoints
        precision="16-mixed",
    )

    # --- Training ---
    print("Starting training...")
    try:
        trainer.fit(model, datamodule=cifar_dm)
    except Exception as e:
        print(f"An error occurred during training: {e}")
        print(
            "Ensure your EncoderNd is correctly implemented and compatible with 2D inputs (3x32x32 for CIFAR-10)."
        )
        print("The placeholder EncoderNd is very basic and might not learn well.")
        raise

    # --- Testing ---
    print("\nStarting testing...")
    try:
        trainer.test(
            model, datamodule=cifar_dm
        )  # Uses the best checkpoint by default if checkpointing is enabled
        # Or test with the last model: trainer.test(datamodule=cifar_dm)
    except Exception as e:
        print(f"An error occurred during testing: {e}")

    print("\nExample finished. Check 'logs/' directory for training logs.")
