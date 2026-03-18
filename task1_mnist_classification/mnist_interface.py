from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class MnistClassifierInterface(ABC):
    """
    Abstract interface for MNIST classifiers.

    All concrete classifiers (Random Forest, Feed-Forward NN, CNN) must implement
    the same training and prediction interface so that they can be used
    interchangeably via the MnistClassifier wrapper.
    """

    @abstractmethod
    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Train the classifier.

        Parameters
        ----------
        x_train:
            Array of training images. Expected shape: (n_samples, 1, 28, 28)
            or (n_samples, 28, 28). The concrete implementation may reshape
            internally if needed.
        y_train:
            Array of integer labels, shape (n_samples,).
        x_val, y_val:
            Optional validation data. Implementations may ignore it.
        kwargs:
            Additional algorithm-specific training hyperparameters.
        """

    @abstractmethod
    def predict(self, x: np.ndarray, batch_size: int | None = None) -> np.ndarray:
        """
        Predict class labels for the given images.

        Parameters
        ----------
        x:
            Array of images to classify. Expected shape: (n_samples, 1, 28, 28)
            or (n_samples, 28, 28).
        batch_size:
            Optional batch size hint for implementations that support
            batched inference (e.g. neural networks). If None, a sensible
            default can be used.

        Returns
        -------
        np.ndarray:
            Array of integer predictions, shape (n_samples,).
        """

