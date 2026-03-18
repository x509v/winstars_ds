## Task 1 – MNIST Image Classification (OOP)

This folder contains the solution for **Task 1** of the test assignment: image classification on the
MNIST dataset using three different algorithms (Random Forest, Feed-Forward Neural Network,
and Convolutional Neural Network) wrapped into a common object-oriented interface.

### Project structure

- `mnist_interface.py` – defines the abstract `MnistClassifierInterface` with `train` and `predict` methods.
- `models.py` – contains three concrete model classes:
  - `RandomForestMnistClassifier`
  - `FeedForwardMnistClassifier`
  - `ConvolutionalMnistClassifier`
- `mnist_classifier.py` – defines the unified wrapper class `MnistClassifier` that hides the concrete implementation behind a single API.
- `requirements.txt` – Python dependencies for this task.
- `demo_task1.ipynb` – Jupyter Notebook demonstrating how to use the models, including edge cases.

### OOP design

- **Interface**: `MnistClassifierInterface` declares two abstract methods:
  - `train(x_train, y_train, x_val=None, y_val=None, **kwargs)`
  - `predict(x, batch_size=None) -> np.ndarray`
- **Concrete models**: each of the three models implements this interface with the same method
  signatures and the same input/output conventions:
  - Images are accepted as NumPy arrays of shape `(n_samples, 1, 28, 28)` or `(n_samples, 28, 28)`.
  - Predictions are returned as a 1D NumPy array of integer class indices of shape `(n_samples,)`.
- **Wrapper**: `MnistClassifier` takes the algorithm name as a constructor parameter and forwards
  all calls to the chosen implementation:

  ```python
  from mnist_classifier import MnistClassifier

  clf = MnistClassifier(algorithm="cnn", epochs=2)
  clf.train(x_train, y_train)
  preds = clf.predict(x_test)
  ```

  Supported values for `algorithm` are:
  - `"rf"` – `RandomForestMnistClassifier`
  - `"nn"` – `FeedForwardMnistClassifier`
  - `"cnn"` – `ConvolutionalMnistClassifier`

### Models

- **RandomForestMnistClassifier**
  - Implemented with `sklearn.ensemble.RandomForestClassifier`.
  - Images are flattened into 1D vectors before training (`28 * 28` features).
  - Suitable as a simple baseline model.

- **FeedForwardMnistClassifier**
  - Implemented with PyTorch as a small fully-connected network:
    - Input layer: 28×28 pixels flattened.
    - Two hidden layers with ReLU activations.
    - Output layer with 10 units (one per digit 0–9).
  - Uses mini-batch training with Adam optimizer and cross-entropy loss.

- **ConvolutionalMnistClassifier**
  - Implemented with PyTorch as a small CNN:
    - Two convolutional layers with ReLU and max-pooling.
    - One fully-connected hidden layer.
    - Output layer with 10 units.
  - Also uses mini-batch training with Adam optimizer and cross-entropy loss.

### Setup and usage

#### 1. Create and activate virtual environment (recommended)

```bash
cd task1_mnist_classification
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
```

#### 2. Install dependencies

```bash
pip install -r requirements.txt
```

#### 3. Run the demo notebook

Start Jupyter and open `demo_task1.ipynb`:

```bash
jupyter notebook
```

Then:
- Open `demo_task1.ipynb`.
- Run the cells in order to:
  - Download the MNIST dataset.
  - Train each of the three models.
  - Evaluate their accuracy on a held-out test set.
  - Inspect predictions and visualize a few samples.

### Edge cases covered in the demo

The notebook demonstrates:

- **Input shape handling**:
  - Passing images as `(n_samples, 28, 28)` and `(n_samples, 1, 28, 28)`.
- **Small batches**:
  - Predicting on very small subsets (including a single image).
- **Invalid shapes**:
  - Example of how a wrong input shape triggers a clear `ValueError`.

### Notes

- Training epochs and model sizes are configured to run quickly on a CPU, but you can increase
  `epochs`, `hidden_dim`, or other hyperparameters if you want higher accuracy.
- If a GPU is available, the neural network models will automatically use CUDA when possible.

