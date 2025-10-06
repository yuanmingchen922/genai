"""
Data Loader Module
Handles data loading for various datasets including Fashion-MNIST.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np


def load_fashion_mnist(batch_size=32, train=True, download=True, root='./data'):
    """
    Load Fashion-MNIST dataset with preprocessing.
    
    Args:
        batch_size (int): Number of samples per batch. Default: 32
        train (bool): If True, load training set; otherwise load test set. Default: True
        download (bool): If True, download the dataset if not available. Default: True
        root (str): Root directory for dataset storage. Default: './data'
    
    Returns:
        DataLoader: PyTorch DataLoader object for the Fashion-MNIST dataset
        
    Example:
        >>> train_loader = load_fashion_mnist(batch_size=64, train=True)
        >>> test_loader = load_fashion_mnist(batch_size=64, train=False)
    """
    # Define preprocessing transformations
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL Image to tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    
    # Load the dataset
    dataset = torchvision.datasets.FashionMNIST(
        root=root,
        train=train,
        download=download,
        transform=transform
    )
    
    # Create and return DataLoader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,  # Shuffle training data
        num_workers=2,  # Use multiple workers for data loading
        pin_memory=True  # Speed up transfer to GPU
    )
    
    return data_loader


def get_dataset_info(dataset_name='fashion_mnist'):
    """
    Get information about the dataset.
    
    Args:
        dataset_name (str): Name of the dataset. Default: 'fashion_mnist'
    
    Returns:
        dict: Dictionary containing dataset information
    """
    if dataset_name == 'fashion_mnist':
        return {
            'name': 'Fashion-MNIST',
            'num_classes': 10,
            'input_shape': (1, 28, 28),
            'classes': [
                'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
            ]
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


__all__ = ['load_fashion_mnist', 'get_dataset_info']
