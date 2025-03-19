"""
Training script for CIFAR-10 classification using Swin Transformer.
"""

import os
import argparse
from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

from model import ViT
from utils import (
    get_cifar10_dataloaders,
    train_one_epoch,
    evaluate,
    plot_training_curves,
    visualize_predictions,
)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train Swin Transformer on CIFAR-10")
    parser.add_argument(
        "--batch-size", type=int, default=512, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs to train"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="Weight decay")
    parser.add_argument(
        "--embed-dim", type=int, default=32, help="Initial embedding dimension"
    )
    parser.add_argument(
        "--depths",
        type=List[int],
        default=[2, 4, 8, 2],
        help="Number of transformer blocks",
    )
    parser.add_argument(
        "--num-heads",
        type=List[int],
        default=[2, 4, 8, 16],
        help="Number of attention heads",
    )
    parser.add_argument("--patch-size", type=int, default=2, help="Patch size")
    parser.add_argument(
        "--reduction-size", type=int, default=2, help="Reduction size for downsampling"
    )
    parser.add_argument("--dropout", type=float, default=0.01, help="Dropout rate")
    parser.add_argument(
        "--output-dir", type=str, default="./output", help="Directory to save outputs"
    )
    args = parser.parse_args()

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"cifar10_swin_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Save configuration
    with open(os.path.join(output_dir, "config.txt"), "w") as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get dataloaders
    trainloader, testloader, classes = get_cifar10_dataloaders(
        batch_size=args.batch_size
    )

    # Initialize model
    model = ViT(
        img_size=32,
        patch_size=args.patch_size,
        reduction_size=args.reduction_size,
        in_channels=3,
        num_classes=10,
        embed_dim=args.embed_dim,
        depths=args.depths,
        num_heads=args.num_heads,
        mlp_ratio=4,
        dropout=args.dropout,
    ).to(device)

    model = torch.compile(
        model,
        backend="eager",
    )

    # Print model summary (parameter count)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {param_count / 1e6:.2f}M parameters")

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Learning rate scheduler (cosine annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss, train_acc = train_one_epoch(
            model, trainloader, criterion, optimizer, device
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Evaluate
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        # Update learning rate
        scheduler.step()

        # Save model checkpoint
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                },
                os.path.join(output_dir, f"checkpoint_epoch_{epoch + 1}.pth"),
            )

    print("Training completed!")

    # Plot training and validation curves
    plot_training_curves(
        train_losses, train_accs, test_losses, test_accs, save_dir=output_dir
    )

    # Visualize predictions
    visualize_predictions(model, testloader, classes, device, save_dir=output_dir)

    # Save the final model
    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pth"))
    print(f"Model saved to {os.path.join(output_dir, 'final_model.pth')}")

    # Save training history
    history = {
        "train_losses": train_losses,
        "train_accs": train_accs,
        "test_losses": test_losses,
        "test_accs": test_accs,
    }
    torch.save(history, os.path.join(output_dir, "training_history.pth"))


if __name__ == "__main__":
    main()
