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


class CustomImageDataset(torch.utils.data.Dataset):
    """
    Custom dataset class for loading images from a directory.
    Useful for training GANs on custom image datasets.
    
    Args:
        root_dir (str): Root directory containing images
        transform (callable, optional): Optional transform to be applied on images
        valid_extensions (tuple): Valid image file extensions. Default: ('.png', '.jpg', '.jpeg', '.bmp')
    
    Example:
        >>> from torchvision import transforms
        >>> transform = transforms.Compose([
        ...     transforms.CenterCrop(178),
        ...     transforms.Resize(64),
        ...     transforms.ToTensor(),
        ...     transforms.Normalize([0.5]*3, [0.5]*3)
        ... ])
        >>> dataset = CustomImageDataset('../data/img_align_celeba', transform=transform)
        >>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    """
    
    def __init__(self, root_dir, transform=None, valid_extensions=('.png', '.jpg', '.jpeg', '.bmp')):
        self.root_dir = root_dir
        self.transform = transform
        self.valid_extensions = valid_extensions
        
        # Get all image paths
        import os
        self.image_paths = [
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.lower().endswith(valid_extensions)
        ]
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {root_dir} with extensions {valid_extensions}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        from PIL import Image
        
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, 0  # dummy label (not used in GANs)


def load_celeba(root_dir, batch_size=128, image_size=64, num_workers=2):
    """
    Load CelebA dataset with GAN-appropriate preprocessing.
    
    Args:
        root_dir (str): Root directory containing CelebA images
        batch_size (int): Batch size for DataLoader. Default: 128
        image_size (int): Target image size (square). Default: 64
        num_workers (int): Number of worker processes. Default: 2
    
    Returns:
        DataLoader: PyTorch DataLoader for CelebA dataset
    
    Example:
        >>> dataloader = load_celeba('../data/img_align_celeba', batch_size=64)
    """
    import torchvision.transforms as transforms
    
    # GAN-specific preprocessing
    transform = transforms.Compose([
        transforms.CenterCrop(178),      # Center crop to remove background
        transforms.Resize(image_size),   # Resize to target size
        transforms.ToTensor(),           # Convert to tensor
        transforms.Normalize([0.5]*3, [0.5]*3),  # Normalize to [-1, 1] for tanh
    ])
    
    dataset = CustomImageDataset(root_dir, transform=transform)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


def create_gan_transforms(image_size=64, normalize_range='tanh'):
    """
    Create transforms suitable for GAN training.
    
    Args:
        image_size (int): Target image size (square). Default: 64
        normalize_range (str): Normalization range ('tanh' for [-1,1] or 'sigmoid' for [0,1]). Default: 'tanh'
    
    Returns:
        torchvision.transforms.Compose: Transform pipeline
    
    Example:
        >>> transform = create_gan_transforms(image_size=128, normalize_range='tanh')
        >>> dataset = CustomImageDataset('../data/images', transform=transform)
    """
    import torchvision.transforms as transforms
    
    if normalize_range == 'tanh':
        # Normalize to [-1, 1] for use with tanh activation
        normalize = transforms.Normalize([0.5]*3, [0.5]*3)
    elif normalize_range == 'sigmoid':
        # Normalize to [0, 1] for use with sigmoid activation  
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    else:
        raise ValueError("normalize_range must be 'tanh' or 'sigmoid'")
    
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])
    
    return transform


__all__ = ['load_fashion_mnist', 'get_dataset_info', 'CustomImageDataset', 'load_celeba', 'create_gan_transforms']
