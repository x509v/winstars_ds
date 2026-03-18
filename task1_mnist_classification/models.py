from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader, TensorDataset

from .mnist_interface import MnistClassifierInterface


def _ensure_channel_first(x: np.ndarray) -> np.ndarray:
    """
    Ensure array has shape (n_samples, 1, 28, 28).
    """
    if x.ndim == 3:
        # (n, 28, 28) -> (n, 1, 28, 28)
        return x[:, None, :, :]
    if x.ndim == 4:
        return x
    raise ValueError(f"Unexpected input shape {x.shape}, expected 3D or 4D array.")


def _to_torch_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x.astype(np.float32))


class RandomForestMnistClassifier(MnistClassifierInterface):
    """
    Random Forest implementation for MNIST.
    """

    def __init__(self, n_estimators: int = 200, random_state: int = 42, **kwargs: Any) -> None:
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state, n_jobs=-1, **kwargs
        )

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        x_train = _ensure_channel_first(x_train)
        n_samples = x_train.shape[0]
        x_flat = x_train.reshape(n_samples, -1)
        self.model.fit(x_flat, y_train)

    def predict(self, x: np.ndarray, batch_size: int | None = None) -> np.ndarray:
        x = _ensure_channel_first(x)
        n_samples = x.shape[0]
        x_flat = x.reshape(n_samples, -1)
        return self.model.predict(x_flat)


class _SimpleFFNN(nn.Module):
    def __init__(self, input_dim: int = 28 * 28, hidden_dim: int = 256, num_classes: int = 10) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


class _SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.out = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.out(x)


@dataclass
class _NNTrainingConfig:
    batch_size: int = 128
    epochs: int = 3
    learning_rate: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class FeedForwardMnistClassifier(MnistClassifierInterface):
    """
    Feed-forward neural network implementation for MNIST.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        batch_size: int = 128,
        epochs: int = 3,
        learning_rate: float = 1e-3,
        device: str | None = None,
    ) -> None:
        self.config = _NNTrainingConfig(
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        )
        self.model = _SimpleFFNN(hidden_dim=hidden_dim).to(self.config.device)

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        x_train = _ensure_channel_first(x_train) / 255.0
        x_tensor = _to_torch_tensor(x_train)
        y_tensor = torch.from_numpy(y_train.astype(np.int64))

        dataset = TensorDataset(x_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for _ in range(self.config.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.config.device)
                batch_y = batch_y.to(self.config.device)

                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

    def predict(self, x: np.ndarray, batch_size: int | None = None) -> np.ndarray:
        x = _ensure_channel_first(x) / 255.0
        x_tensor = _to_torch_tensor(x)
        device = self.config.device

        bsz = batch_size or self.config.batch_size
        loader = DataLoader(TensorDataset(x_tensor), batch_size=bsz, shuffle=False)

        self.model.eval()
        preds: list[int] = []
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(device)
                logits = self.model(batch_x)
                batch_preds = torch.argmax(logits, dim=1)
                preds.extend(batch_preds.cpu().numpy().tolist())
        return np.array(preds, dtype=np.int64)


class ConvolutionalMnistClassifier(MnistClassifierInterface):
    """
    Convolutional neural network implementation for MNIST.
    """

    def __init__(
        self,
        batch_size: int = 128,
        epochs: int = 3,
        learning_rate: float = 1e-3,
        device: str | None = None,
    ) -> None:
        self.config = _NNTrainingConfig(
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        )
        self.model = _SimpleCNN().to(self.config.device)

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        x_train = _ensure_channel_first(x_train) / 255.0
        x_tensor = _to_torch_tensor(x_train)
        y_tensor = torch.from_numpy(y_train.astype(np.int64))

        dataset = TensorDataset(x_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for _ in range(self.config.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.config.device)
                batch_y = batch_y.to(self.config.device)

                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

    def predict(self, x: np.ndarray, batch_size: int | None = None) -> np.ndarray:
        x = _ensure_channel_first(x) / 255.0
        x_tensor = _to_torch_tensor(x)
        device = self.config.device

        bsz = batch_size or self.config.batch_size
        loader = DataLoader(TensorDataset(x_tensor), batch_size=bsz, shuffle=False)

        self.model.eval()
        preds: list[int] = []
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(device)
                logits = self.model(batch_x)
                batch_preds = torch.argmax(logits, dim=1)
                preds.extend(batch_preds.cpu().numpy().tolist())
        return np.array(preds, dtype=np.int64)

