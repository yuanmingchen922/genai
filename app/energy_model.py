"""
Energy-Based Model Implementation for CIFAR-10
Uses Langevin Dynamics for sampling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class EnergyNetwork(nn.Module):
    """
    Energy function network that assigns scalar energy to images.
    Lower energy = more realistic image.
    """
    
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        
        # Convolutional layers to extract features
        self.conv1 = nn.Conv2d(in_channels, base_channels, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(base_channels * 4, base_channels * 8, 3, stride=2, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.bn2 = nn.BatchNorm2d(base_channels * 2)
        self.bn3 = nn.BatchNorm2d(base_channels * 4)
        self.bn4 = nn.BatchNorm2d(base_channels * 8)
        
        # Global pooling and energy output
        # For CIFAR-10 (32x32), after 4 stride-2 convs: 32 -> 16 -> 8 -> 4 -> 2
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.energy_head = nn.Linear(base_channels * 8, 1)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) input images
        
        Returns:
            (B, 1) energy values (scalar per image)
        """
        # Feature extraction
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Energy output (scalar)
        energy = self.energy_head(x)
        
        return energy


class SpectralNormConv2d(nn.Module):
    """Convolutional layer with spectral normalization for stable training."""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.utils.spectral_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        )
    
    def forward(self, x):
        return self.conv(x)


class ImprovedEnergyNetwork(nn.Module):
    """
    Improved Energy Network with spectral normalization and residual connections.
    More stable training and better gradient properties.
    """
    
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        
        # Use spectral normalization for stable gradients
        self.conv1 = SpectralNormConv2d(in_channels, base_channels, 3, stride=1, padding=1)
        self.conv2 = SpectralNormConv2d(base_channels, base_channels, 3, stride=2, padding=1)
        
        self.conv3 = SpectralNormConv2d(base_channels, base_channels * 2, 3, stride=1, padding=1)
        self.conv4 = SpectralNormConv2d(base_channels * 2, base_channels * 2, 3, stride=2, padding=1)
        
        self.conv5 = SpectralNormConv2d(base_channels * 2, base_channels * 4, 3, stride=1, padding=1)
        self.conv6 = SpectralNormConv2d(base_channels * 4, base_channels * 4, 3, stride=2, padding=1)
        
        # Global pooling and energy head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.energy_head = nn.utils.spectral_norm(nn.Linear(base_channels * 4, 1))
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) input images, normalized to [-1, 1]
        
        Returns:
            (B, 1) energy values
        """
        # First block
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        
        # Second block
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        
        # Third block
        x = F.leaky_relu(self.conv5(x), 0.2)
        x = F.leaky_relu(self.conv6(x), 0.2)
        
        # Energy output
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        energy = self.energy_head(x)
        
        return energy


class EnergyModel:
    """
    Complete Energy-Based Model with Langevin sampling.
    
    Training:
    - Maximize energy for real images
    - Minimize energy for generated samples (via Langevin dynamics)
    
    Sampling:
    - Start from random noise
    - Iteratively follow negative energy gradient (Langevin dynamics)
    """
    
    def __init__(
        self,
        model: nn.Module,
        device='cpu',
        langevin_steps=60,
        langevin_lr=0.01,
        langevin_noise=0.005
    ):
        self.model = model
        self.device = device
        self.langevin_steps = langevin_steps
        self.langevin_lr = langevin_lr
        self.langevin_noise = langevin_noise
    
    def langevin_sample(
        self,
        x_init: Optional[torch.Tensor] = None,
        batch_size: int = 16,
        channels: int = 3,
        height: int = 32,
        width: int = 32,
        return_trajectory: bool = False
    ) -> torch.Tensor:
        """
        Generate samples using Langevin Dynamics.
        
        Langevin Dynamics update:
        x_{t+1} = x_t - lr/2 * ∇_x E(x_t) + N(0, lr*σ^2)
        
        Where:
        - E(x): energy function
        - ∇_x E(x): gradient of energy w.r.t. input
        - lr: learning rate (step size)
        - σ: noise standard deviation
        
        Args:
            x_init: optional initial samples
            batch_size, channels, height, width: sample dimensions
            return_trajectory: if True, return all intermediate samples
        
        Returns:
            (B, C, H, W) generated samples
            or list of intermediate samples if return_trajectory=True
        """
        self.model.eval()
        
        # Initialize from noise if not provided
        if x_init is None:
            x = torch.randn(batch_size, channels, height, width).to(self.device)
        else:
            x = x_init.clone()
        
        # Normalize to [-1, 1]
        x = torch.clamp(x, -1, 1)
        x.requires_grad = True
        
        trajectory = [x.detach().clone()] if return_trajectory else None
        
        # Langevin dynamics
        for step in range(self.langevin_steps):
            # Compute energy
            energy = self.model(x).sum()
            
            # Compute gradient of energy w.r.t. input
            energy.backward()
            
            with torch.no_grad():
                # Langevin update: move toward lower energy
                x_grad = x.grad
                x = x - self.langevin_lr / 2 * x_grad
                
                # Add noise
                noise = torch.randn_like(x) * self.langevin_noise
                x = x + noise
                
                # Clip to valid range
                x = torch.clamp(x, -1, 1)
            
            # Reset gradient
            x = x.detach()
            x.requires_grad = True
            
            if return_trajectory:
                trajectory.append(x.detach().clone())
        
        x = x.detach()
        
        if return_trajectory:
            return trajectory
        else:
            return x
    
    def contrastive_divergence_loss(
        self,
        real_images: torch.Tensor,
        num_neg_samples: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate contrastive divergence loss for training.
        
        Loss = E(x_real) - E(x_fake)
        
        We want to:
        - Minimize energy for real images
        - Maximize energy (push up) for generated samples
        
        Args:
            real_images: (B, C, H, W) real images
            num_neg_samples: number of negative samples (default: same as batch size)
        
        Returns:
            loss: scalar loss value
            fake_images: generated negative samples
        """
        batch_size = real_images.size(0)
        if num_neg_samples is None:
            num_neg_samples = batch_size
        
        # Energy for real images (minimize this)
        energy_real = self.model(real_images)
        
        # Generate negative samples via Langevin sampling
        with torch.no_grad():
            fake_images = self.langevin_sample(
                batch_size=num_neg_samples,
                channels=real_images.size(1),
                height=real_images.size(2),
                width=real_images.size(3)
            )
        
        # Energy for fake images (maximize this, i.e., minimize negative)
        energy_fake = self.model(fake_images)
        
        # Contrastive divergence loss
        # We want: low energy for real, high energy for fake
        # So minimize: E(real) - E(fake)
        loss = energy_real.mean() - energy_fake.mean()
        
        return loss, fake_images
    
    def train_step(
        self,
        real_images: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> dict:
        """
        Perform one training step.
        
        Args:
            real_images: (B, C, H, W) real images
            optimizer: optimizer for model parameters
        
        Returns:
            dict with loss and energy statistics
        """
        self.model.train()
        optimizer.zero_grad()
        
        # Calculate contrastive divergence loss
        loss, fake_images = self.contrastive_divergence_loss(real_images)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        with torch.no_grad():
            energy_real = self.model(real_images).mean().item()
            energy_fake = self.model(fake_images).mean().item()
        
        return {
            'loss': loss.item(),
            'energy_real': energy_real,
            'energy_fake': energy_fake,
            'energy_gap': energy_real - energy_fake
        }
    
    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        channels: int = 3,
        height: int = 32,
        width: int = 32
    ) -> torch.Tensor:
        """
        Generate samples for inference.
        
        Args:
            num_samples: number of samples to generate
            channels, height, width: image dimensions
        
        Returns:
            (num_samples, C, H, W) generated images
        """
        return self.langevin_sample(
            batch_size=num_samples,
            channels=channels,
            height=height,
            width=width,
            return_trajectory=False
        )


def get_energy_model(device='cpu', use_improved=True):
    """Create and return an Energy Model instance."""
    if use_improved:
        energy_net = ImprovedEnergyNetwork(
            in_channels=3,
            base_channels=64
        ).to(device)
    else:
        energy_net = EnergyNetwork(
            in_channels=3,
            base_channels=64
        ).to(device)
    
    energy_model = EnergyModel(
        model=energy_net,
        device=device,
        langevin_steps=60,
        langevin_lr=0.01,
        langevin_noise=0.005
    )
    
    return energy_model


# Helper function for gradient calculation (used in theory questions)
def compute_energy_gradient(model, x):
    """
    Compute gradient of energy function with respect to input.
    
    This is used in Langevin sampling:
    ∇_x E(x) = ∂E/∂x
    
    Args:
        model: energy model
        x: input tensor with requires_grad=True
    
    Returns:
        gradient tensor
    """
    x.requires_grad = True
    energy = model(x).sum()
    energy.backward()
    grad = x.grad
    x.grad = None
    return grad

