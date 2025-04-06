import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from lightning.fabric import Fabric
from tqdm import tqdm

from model import ShiftingWindowTransformer


def main() -> None:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    batch_size = 128

    trainset, valset = torch.utils.data.random_split(
        torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        ),
        [0.7, 0.3],
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    valloader = torch.utils.data.DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    network: nn.Module = ShiftingWindowTransformer(
        img_size=32,
        patch_size=(4, 4, 4),
        reduction_size=(2, 2, 2),
        in_channels=3,
        num_classes=len(classes),
        embed_dim=128,
        depths=(2, 6, 4),
        num_heads=(2, 4, 8),
        mlp_ratio=3,
        dropout=0.05,
    )
    # network = torch.compile(network)

    criterion: nn.Module = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer: optim.Optimizer = optim.AdamW(network.parameters(), lr=3e-4)

    fabric = Fabric()
    fabric.launch()

    network, optimizer = fabric.setup(network, optimizer)
    trainloader, valloader = fabric.setup_dataloaders(trainloader, valloader)

    epoch_total = 20
    training_acc = []
    training_loss = []
    validation_acc = []
    validation_loss = []
    for epoch in range(epoch_total):
        if fabric.is_global_zero:
            pbar = tqdm(trainloader, desc="Training", total=len(trainloader))

        network.train()
        running_loss = 0
        total_correct = 0
        total_samples = 0
        for i, batch in enumerate(pbar):
            inputs: torch.Tensor
            targets: torch.Tensor
            inputs, targets = batch

            optimizer.zero_grad()

            outputs = network(inputs)
            loss: torch.Tensor = criterion(outputs, targets)

            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == targets).sum().item()
            total_correct += correct
            total_samples += targets.size(0)

            fabric.backward(loss)
            optimizer.step()

            if fabric.is_global_zero:
                running_loss = running_loss + (
                    (1.0 / (i + 1)) * (loss.item() - running_loss)
                )
                running_acc = total_correct / total_samples
                pbar.set_postfix({"loss": running_loss, "acc": running_acc})

        if fabric.is_global_zero:
            pbar = tqdm(valloader, desc="Validation", total=len(valloader))
            training_acc.append(running_acc)
            training_loss.append(running_loss)

        network.eval()
        running_loss = 0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for i, batch in enumerate(pbar):
                inputs: torch.Tensor
                targets: torch.Tensor
                inputs, targets = batch

                outputs = network(inputs)
                loss: torch.Tensor = criterion(outputs, targets)

                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == targets).sum().item()
                total_correct += correct
                total_samples += targets.size(0)

                if fabric.is_global_zero:
                    running_loss = running_loss + (
                        (1.0 / (i + 1)) * (loss.item() - running_loss)
                    )
                    running_acc = total_correct / total_samples
                    pbar.set_postfix({"loss": running_loss, "acc": running_acc})
                    pbar.update(1)

        if fabric.is_global_zero:
            validation_acc.append(running_acc)
            validation_loss.append(running_loss)
            print(
                f"Epoch {epoch + 1} - Train Loss: {training_loss[-1]:.4f}, Train Acc: {training_acc[-1]:.4f}, "
                f"Val Loss: {validation_loss[-1]:.4f}, Val Acc: {validation_acc[-1]:.4f}"
            )


if __name__ == "__main__":
    main()
