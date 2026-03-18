from __future__ import annotations

from typing import Any

import numpy as np

from .mnist_interface import MnistClassifierInterface
from .models import (
    ConvolutionalMnistClassifier,
    FeedForwardMnistClassifier,
    RandomForestMnistClassifier,
)


class MnistClassifier:
    """
    Unified entry point for MNIST classification.

    This wrapper hides the concrete algorithm implementation behind a single
    interface so that the user interacts with the same methods regardless of
    whether a Random Forest, feed-forward NN, or CNN is used.
    """

    def __init__(self, algorithm: str, **model_kwargs: Any) -> None:
        """
        Parameters
        ----------
        algorithm:
            Which model to use. Supported values:
            - \"rf\"  : RandomForestMnistClassifier
            - \"nn\"  : FeedForwardMnistClassifier
            - \"cnn\" : ConvolutionalMnistClassifier
        model_kwargs:
            Extra keyword arguments propagated to the concrete model
            constructor (e.g. number of estimators, learning rate, etc.).
        """
        algorithm = algorithm.lower()
        if algorithm == "rf":
            self._model: MnistClassifierInterface = RandomForestMnistClassifier(**model_kwargs)
        elif algorithm == "nn":
            self._model = FeedForwardMnistClassifier(**model_kwargs)
        elif algorithm == "cnn":
            self._model = ConvolutionalMnistClassifier(**model_kwargs)
        else:
            raise ValueError(f"Unsupported algorithm '{algorithm}'. Use 'rf', 'nn', or 'cnn'.")

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Train the underlying model.

        This method forwards all arguments to the concrete model's train method,
        ensuring a unified external API.
        """
        self._model.train(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, **kwargs)

    def predict(self, x: np.ndarray, batch_size: int | None = None) -> np.ndarray:
        """
        Predict class labels for the given images.

        Parameters
        ----------
        x:
            Array of images to classify. Expected shape: (n_samples, 1, 28, 28)
            or (n_samples, 28, 28).
        batch_size:
            Optional batch size passed to neural-network implementations.
            The Random Forest implementation ignores this parameter.

        Returns
        -------
        np.ndarray:
            Array of integer predictions, shape (n_samples,).
        """
        return self._model.predict(x=x, batch_size=batch_size)

