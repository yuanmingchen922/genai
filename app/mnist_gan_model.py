"""
MNIST GAN Model for hand-written digit generation
Implements a GAN trained on MNIST dataset for generating synthetic digits
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Dict, Optional
import base64
from io import BytesIO
from PIL import Image


class MNISTGenerator(nn.Module):
    """
    Generator Network for MNIST GAN.
    
    Architecture:
    - Input: Noise vector (batch_size, 100)
    - FC layer: 100 -> 7*7*128
    - Reshape to (batch_size, 128, 7, 7)
    - ConvTranspose2d: 128 -> 64, 14x14
    - ConvTranspose2d: 64 -> 1, 28x28
    """
    
    def __init__(self, noise_dim=100):
        super(MNISTGenerator, self).__init__()
        self.noise_dim = noise_dim
        
        # Fully connected layer: 100 -> 7*7*128
        self.fc = nn.Linear(noise_dim, 7 * 7 * 128)
        
        # Transpose convolution layers
        self.conv_transpose1 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1
        )  # Output: 64 x 14 x 14
        
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv_transpose2 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=1,
            kernel_size=4,
            stride=2,
            padding=1
        )  # Output: 1 x 28 x 28
        
        self.tanh = nn.Tanh()
    
    def forward(self, z):
        """
        Forward pass.
        
        Args:
            z: Noise vector of shape (batch_size, 100)
        
        Returns:
            Generated images of shape (batch_size, 1, 28, 28)
        """
        # FC layer and reshape
        x = self.fc(z)  # (batch_size, 7*7*128)
        x = x.view(-1, 128, 7, 7)  # (batch_size, 128, 7, 7)
        
        # First transpose convolution block
        x = self.conv_transpose1(x)  # (batch_size, 64, 14, 14)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Second transpose convolution block
        x = self.conv_transpose2(x)  # (batch_size, 1, 28, 28)
        x = self.tanh(x)  # Output range: [-1, 1]
        
        return x


class MNISTDiscriminator(nn.Module):
    """
    Discriminator Network for MNIST GAN.
    
    Architecture:
    - Input: Images (batch_size, 1, 28, 28)
    - Conv2d: 1 -> 64, 14x14
    - Conv2d: 64 -> 128, 7x7
    - Flatten and FC to single output
    """
    
    def __init__(self):
        super(MNISTDiscriminator, self).__init__()
        
        # First convolution layer
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1
        )  # Output: 64 x 14 x 14
        
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        
        # Second convolution layer
        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=4,
            stride=2,
            padding=1
        )  # Output: 128 x 7 x 7
        
        self.bn2 = nn.BatchNorm2d(128)
        
        # Fully connected layer
        self.fc = nn.Linear(128 * 7 * 7, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input images of shape (batch_size, 1, 28, 28)
        
        Returns:
            Predictions of shape (batch_size, 1), range [0, 1]
        """
        # First convolution block
        x = self.conv1(x)  # (batch_size, 64, 14, 14)
        x = self.leaky_relu(x)
        
        # Second convolution block
        x = self.conv2(x)  # (batch_size, 128, 7, 7)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        
        # Flatten and FC layer
        x = x.view(-1, 128 * 7 * 7)  # (batch_size, 128*7*7)
        x = self.fc(x)  # (batch_size, 1)
        x = self.sigmoid(x)
        
        return x


class MNISTGANGenerator:
    """
    MNIST GAN Generator Service for API integration.
    Handles model loading, image generation, and format conversion.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the MNIST GAN Generator.
        
        Args:
            model_path: Path to the trained generator model
            device: Device to run the model on ('cuda', 'mps', 'cpu', or None for auto)
        """
        # Set device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        self.noise_dim = 100
        self.generator = MNISTGenerator(noise_dim=self.noise_dim).to(self.device)
        
        # Load trained model if path provided
        if model_path:
            self.load_model(model_path)
        
        self.generator.eval()
    
    def load_model(self, model_path: str):
        """
        Load trained generator weights.
        
        Args:
            model_path: Path to the saved model checkpoint
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.generator.load_state_dict(checkpoint['model_state_dict'])
                elif 'generator_state_dict' in checkpoint:
                    self.generator.load_state_dict(checkpoint['generator_state_dict'])
                else:
                    self.generator.load_state_dict(checkpoint)
            else:
                self.generator.load_state_dict(checkpoint)
            
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Warning: Could not load model from {model_path}: {str(e)}")
            print("Using randomly initialized weights")
    
    def generate_images(
        self, 
        num_images: int = 1, 
        noise: Optional[torch.Tensor] = None,
        return_format: str = 'tensor'
    ):
        """
        Generate synthetic MNIST digits.
        
        Args:
            num_images: Number of images to generate
            noise: Optional custom noise vector. If None, random noise is used.
            return_format: Format of returned images ('tensor', 'numpy', 'pil', 'base64')
        
        Returns:
            Generated images in the specified format
        """
        self.generator.eval()
        
        with torch.no_grad():
            # Generate or use provided noise
            if noise is None:
                noise = torch.randn(num_images, self.noise_dim).to(self.device)
            else:
                noise = noise.to(self.device)
            
            # Generate images
            generated_images = self.generator(noise)
            
            # Denormalize from [-1, 1] to [0, 1]
            generated_images = (generated_images + 1) / 2
            
            # Convert to requested format
            if return_format == 'tensor':
                return generated_images.cpu()
            elif return_format == 'numpy':
                return generated_images.cpu().numpy()
            elif return_format == 'pil':
                return self._tensor_to_pil(generated_images.cpu())
            elif return_format == 'base64':
                return self._tensor_to_base64(generated_images.cpu())
            else:
                raise ValueError(f"Unknown return_format: {return_format}")
    
    def _tensor_to_pil(self, tensor: torch.Tensor) -> List[Image.Image]:
        """Convert tensor to PIL Images."""
        images = []
        for i in range(tensor.size(0)):
            img_tensor = tensor[i].squeeze()  # Remove channel dimension
            img_array = (img_tensor.numpy() * 255).astype(np.uint8)
            img = Image.fromarray(img_array, mode='L')
            images.append(img)
        return images
    
    def _tensor_to_base64(self, tensor: torch.Tensor) -> List[str]:
        """Convert tensor to base64 encoded strings."""
        pil_images = self._tensor_to_pil(tensor)
        base64_images = []
        
        for img in pil_images:
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            base64_images.append(img_str)
        
        return base64_images
    
    def generate_digit(self, digit: Optional[int] = None, seed: Optional[int] = None) -> str:
        """
        Generate a single digit image.
        
        Args:
            digit: Specific digit to generate (0-9). If None, random generation.
            seed: Random seed for reproducibility
        
        Returns:
            Base64 encoded PNG image
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        # Note: Without conditional GAN, we can't guarantee a specific digit
        # This is a simple generation that returns a random digit
        base64_images = self.generate_images(num_images=1, return_format='base64')
        return base64_images[0]
    
    def generate_batch(self, batch_size: int = 16, grid: bool = True) -> str:
        """
        Generate a batch of digit images.
        
        Args:
            batch_size: Number of images to generate
            grid: If True, return images arranged in a grid
        
        Returns:
            Base64 encoded PNG image (grid or list)
        """
        if grid:
            # Generate images as a grid
            images = self.generate_images(num_images=batch_size, return_format='tensor')
            grid_img = self._make_grid(images, nrow=int(np.sqrt(batch_size)))
            return grid_img
        else:
            # Return list of individual images
            return self.generate_images(num_images=batch_size, return_format='base64')
    
    def _make_grid(self, tensor: torch.Tensor, nrow: int = 8, padding: int = 2) -> str:
        """
        Create a grid of images and return as base64.
        
        Args:
            tensor: Tensor of images (N, C, H, W)
            nrow: Number of images per row
            padding: Padding between images
        
        Returns:
            Base64 encoded grid image
        """
        from torchvision.utils import make_grid
        
        grid = make_grid(tensor, nrow=nrow, padding=padding, normalize=False)
        grid_np = grid.numpy().transpose(1, 2, 0)
        
        # Convert to uint8
        grid_np = (grid_np * 255).astype(np.uint8)
        
        # Handle grayscale
        if grid_np.shape[2] == 1:
            grid_np = grid_np.squeeze(2)
        
        # Convert to PIL and then base64
        grid_img = Image.fromarray(grid_np, mode='L' if len(grid_np.shape) == 2 else 'RGB')
        buffered = BytesIO()
        grid_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return img_str
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in self.generator.parameters())
        trainable_params = sum(p.numel() for p in self.generator.parameters() if p.requires_grad)
        
        return {
            "model_type": "MNIST GAN Generator",
            "noise_dimension": self.noise_dim,
            "output_size": "28x28",
            "output_channels": 1,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(self.device)
        }


# Global instance (will be initialized when needed)
_mnist_gan_generator = None


def get_mnist_gan_generator(model_path: Optional[str] = None) -> MNISTGANGenerator:
    """
    Get or create the global MNIST GAN generator instance.
    
    Args:
        model_path: Path to the trained model (optional)
    
    Returns:
        MNISTGANGenerator instance
    """
    global _mnist_gan_generator
    
    if _mnist_gan_generator is None:
        _mnist_gan_generator = MNISTGANGenerator(model_path=model_path)
    
    return _mnist_gan_generator
