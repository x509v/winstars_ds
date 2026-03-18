from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


@dataclass
class ImageTrainConfig:
    data_dir: str
    output_path: str
    batch_size: int = 32
    num_epochs: int = 5
    learning_rate: float = 1e-3
    num_workers: int = 4
    use_pretrained: bool = True


def build_dataloaders(cfg: ImageTrainConfig) -> tuple[DataLoader, DataLoader, Dict[str, int]]:
    """
    Use an ImageFolder dataset with train/val subfolders.

    Expected directory structure:
        data_dir/
            train/
                cat/
                dog/
                ...
            val/
                cat/
                dog/
                ...
    """
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    train_dataset = datasets.ImageFolder(root=f"{cfg.data_dir}/train", transform=train_transform)
    val_dataset = datasets.ImageFolder(root=f"{cfg.data_dir}/val", transform=val_transform)

    train_loader = DataLoader[Any](
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader[Any](
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    return train_loader, val_loader, train_dataset.class_to_idx


def build_model(num_classes: int, use_pretrained: bool = True) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if use_pretrained else None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train_image_classifier(cfg: ImageTrainConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, class_to_idx = build_dataloaders(cfg)

    model = build_model(num_classes=len(class_to_idx), use_pretrained=cfg.use_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    for epoch in range(cfg.num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{cfg.num_epochs} - train loss: {epoch_loss:.4f}")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        if total > 0:
            acc = correct / total
            print(f"Validation accuracy: {acc:.4f}")

    save_obj = {
        "model_state_dict": model.state_dict(),
        "class_to_idx": class_to_idx,
    }
    torch.save(save_obj, cfg.output_path)
    print(f"Model saved to {cfg.output_path}")


def parse_args() -> ImageTrainConfig:
    parser = argparse.ArgumentParser(description="Train an animal image classification model.")
    parser.add_argument("--data-dir", type=str, required=True, help="Root directory with train/val subfolders.")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save trained model (.pt).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-pretrained", action="store_true", help="Disable pretrained ImageNet weights.")
    args = parser.parse_args()
    return ImageTrainConfig(
        data_dir=args.data_dir,
        output_path=args.output_path,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        use_pretrained=not args.no_pretrained,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train_image_classifier(cfg)

