"""
Utility functions for CIFAR-10 training and evaluation
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os


def get_cifar10_dataloaders(batch_size=128, num_workers=2):
    """
    Create DataLoaders for the CIFAR-10 dataset.

    Args:
        batch_size (int): Batch size for training and evaluation.
        num_workers (int): Number of workers for DataLoader.

    Returns:
        tuple: (trainloader, testloader, classes)
    """
    # Data transformations
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # CIFAR-10 classes
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

    return trainloader, testloader, classes


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.

    Args:
        model: The model to train.
        dataloader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to use for training (cuda or cpu).

    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(dataloader, desc="Training")
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update the progress bar
        loop.set_postfix(loss=running_loss / len(loop), acc=100.0 * correct / total)

    return running_loss / len(dataloader), 100.0 * correct / total


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model.

    Args:
        model: The model to evaluate.
        dataloader: DataLoader for evaluation data.
        criterion: Loss function.
        device: Device to use for evaluation (cuda or cpu).

    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(dataloader), 100.0 * correct / total


def plot_training_curves(
    train_losses, train_accs, test_losses, test_accs, save_dir="."
):
    """
    Plot and save training and validation curves.

    Args:
        train_losses (list): Training losses.
        train_accs (list): Training accuracies.
        test_losses (list): Validation losses.
        test_accs (list): Validation accuracies.
        save_dir (str): Directory to save the plot.
    """
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(test_accs, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.tight_layout()

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    plt.savefig(os.path.join(save_dir, "training_curves.png"))


def visualize_predictions(
    model, dataloader, classes, device, num_images=25, save_dir="."
):
    """
    Visualize model predictions on a sample of images.

    Args:
        model: The trained model.
        dataloader: DataLoader for test data.
        classes (list): List of class names.
        device: Device to use for evaluation.
        num_images (int): Number of images to visualize.
        save_dir (str): Directory to save the visualization.
    """
    model.eval()

    # Get a batch of images
    dataiter = iter(dataloader)
    images, labels = next(dataiter)

    # Make predictions
    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    # Move tensors to CPU for visualization
    images = images.cpu()
    labels = labels.cpu()
    predicted = predicted.cpu()

    # Plot the images and their predictions
    fig = plt.figure(figsize=(15, 15))

    for idx in range(min(num_images, len(images))):
        ax = fig.add_subplot(5, 5, idx + 1, xticks=[], yticks=[])

        # Denormalize the image for display
        img = images[idx].numpy().transpose((1, 2, 0))
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2470, 0.2435, 0.2616])
        img = std * img + mean
        img = np.clip(img, 0, 1)

        ax.imshow(img)

        # Set the title color (green for correct, red for incorrect)
        title_color = "green" if predicted[idx] == labels[idx] else "red"
        ax.set_title(f"{classes[predicted[idx]]}", color=title_color)

    plt.tight_layout()

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    plt.savefig(os.path.join(save_dir, "predictions.png"))
