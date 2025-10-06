# Helper Library for Neural Network Projects

A modular and reusable helper library for training and evaluating neural networks with PyTorch.

## Directory Structure

```
helper/
├── __init__.py           # Package initialization
├── data_loader/          # Data loading utilities
│   └── __init__.py      # Fashion-MNIST loader
├── model/               # Neural network models
│   └── __init__.py      # SimpleNN and SimpleCNN
├── trainer/             # Training functions
│   └── __init__.py      # train_model function
├── evaluator/           # Evaluation functions
│   └── __init__.py      # evaluate_model function
└── example_usage.py     # Complete usage example
```

## Quick Start

### 1. Installation

Make sure you have the required dependencies:

```bash
uv add torch torchvision scikit-learn tqdm
```

### 2. Basic Usage

```python
import torch
import torch.nn as nn
import torch.optim as optim

from helper.data_loader import load_fashion_mnist
from helper.model import SimpleNN
from helper.trainer import train_model
from helper.evaluator import evaluate_model

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
train_loader = load_fashion_mnist(batch_size=64, train=True)
test_loader = load_fashion_mnist(batch_size=64, train=False)

# Create model
model = SimpleNN([784, 256, 128, 10], activation='relu', dropout_rate=0.2)
model.to(device)

# Train
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
history = train_model(model, train_loader, criterion, optimizer, device, epochs=5)

# Evaluate
results = evaluate_model(model, test_loader, criterion, device)
print(f"Test Accuracy: {results['accuracy']:.2f}%")
```

## Modules

### Data Loader Module

**File:** `helper/data_loader/__init__.py`

Functions:
- `load_fashion_mnist(batch_size=32, train=True)`: Load Fashion-MNIST dataset
- `get_dataset_info(dataset_name)`: Get dataset metadata

Example:
```python
from helper.data_loader import load_fashion_mnist, get_dataset_info

train_loader = load_fashion_mnist(batch_size=64, train=True)
info = get_dataset_info('fashion_mnist')
print(info['classes'])  # ['T-shirt/top', 'Trouser', ...]
```

### Model Module

**File:** `helper/model/__init__.py`

Classes:
- `SimpleNN(layer_sizes, activation='relu', dropout_rate=0.0)`: Fully connected neural network
- `SimpleCNN(num_classes=10, input_channels=1)`: Convolutional neural network

Example:
```python
from helper.model import SimpleNN, SimpleCNN

# Fully connected network
model1 = SimpleNN([784, 256, 128, 10])

# Convolutional network
model2 = SimpleCNN(num_classes=10, input_channels=1)
```

### Trainer Module

**File:** `helper/trainer/__init__.py`

Functions:
- `train_model(model, train_loader, criterion, optimizer, device, epochs=10)`: Train model
- `train_epoch(model, train_loader, criterion, optimizer, device)`: Train one epoch

Example:
```python
from helper.trainer import train_model

history = train_model(
    model=model,
    train_loader=train_loader,
    criterion=nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model.parameters()),
    device=device,
    epochs=5
)
```

### Evaluator Module

**File:** `helper/evaluator/__init__.py`

Functions:
- `evaluate_model(model, test_loader, criterion, device)`: Comprehensive evaluation
- `calculate_accuracy(model, data_loader, device)`: Calculate accuracy only
- `get_predictions(model, data_loader, device)`: Get predictions and labels

Example:
```python
from helper.evaluator import evaluate_model

results = evaluate_model(model, test_loader, criterion, device)
print(f"Accuracy: {results['accuracy']:.2f}%")
print(f"Precision: {results['precision']:.4f}")
print(f"Recall: {results['recall']:.4f}")
```

## Complete Example

See `helper/example_usage.py` for a complete workflow example:

```bash
python helper/example_usage.py
```

This example demonstrates:
1. Setting up the device (CPU/GPU)
2. Loading Fashion-MNIST dataset
3. Creating a neural network model
4. Training the model with progress tracking
5. Evaluating on test set
6. Saving the trained model
7. Making predictions with the trained model

## Features

- **Modular Design**: Separate concerns into data loading, modeling, training, and evaluation
- **Flexible Models**: Support for both fully connected and convolutional networks
- **Progress Tracking**: Training progress with tqdm progress bars
- **Comprehensive Metrics**: Loss, accuracy, precision, recall, F1-score, confusion matrix
- **GPU Support**: Automatic device selection (CUDA/CPU)
- **Easy to Extend**: Add new datasets, models, or metrics easily

## Advanced Usage

### Custom Model Architecture

```python
from helper.model import SimpleNN

# Create custom architecture
model = SimpleNN(
    layer_sizes=[784, 512, 256, 128, 10],  # Deep network
    activation='relu',
    dropout_rate=0.3  # Higher dropout for regularization
)
```

### Training with Custom Parameters

```python
from helper.trainer import train_model

history = train_model(
    model=model,
    train_loader=train_loader,
    criterion=nn.CrossEntropyLoss(),
    optimizer=torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
    device=device,
    epochs=20,
    verbose=True
)
```

### Get Detailed Predictions

```python
from helper.evaluator import get_predictions

predictions, labels = get_predictions(model, test_loader, device)
print(f"Predictions shape: {predictions.shape}")
print(f"Labels shape: {labels.shape}")
```

## Notes

- The library uses PyTorch as the deep learning framework
- Fashion-MNIST dataset is automatically downloaded on first use
- Models are saved in PyTorch's `.pth` format
- All functions include comprehensive docstrings and examples

## Learning Resources

This helper library is designed based on best practices from:
- Module 4: Modern Deep Learning Architectures Part 1
- Module 5: Modern Deep Learning Architectures Part 2

Key concepts covered:
- Fully Connected Neural Networks (FCNN)
- Convolutional Neural Networks (CNN)
- Proper data loading with DataLoader
- Training loops with gradient descent
- Model evaluation metrics
