"""
Generator Module
Provides image generation functions for VAE and other generative models.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


def generate_samples(model, num_samples=10, latent_dim=2, device='cpu', return_tensor=False):
    """
    Generate new samples using a trained VAE model.
    
    Args:
        model: Trained VAE model with a decoder
        num_samples (int): Number of samples to generate. Default: 10
        latent_dim (int): Dimension of the latent space. Default: 2
        device (str or torch.device): Device to generate on. Default: 'cpu'
        return_tensor (bool): If True, return torch tensor; otherwise numpy array. Default: False
    
    Returns:
        torch.Tensor or np.ndarray: Generated images
    
    Example:
        >>> vae = VAE(latent_dim=2)
        >>> samples = generate_samples(vae, num_samples=16, device='cuda')
    """
    model.eval()
    with torch.no_grad():
        # Sample from standard normal distribution
        z = torch.randn(num_samples, latent_dim).to(device)
        
        # Generate images using decoder
        if hasattr(model, 'decoder'):
            samples = model.decoder(z)
        else:
            samples = model(z)
        
        if return_tensor:
            return samples
        else:
            return samples.cpu().numpy()


def generate_grid(model, grid_size=10, latent_dim=2, device='cpu', z_range=(-3, 3)):
    """
    Generate a grid of images by sampling uniformly in 2D latent space.
    Only works for latent_dim=2.
    
    Args:
        model: Trained VAE model with a decoder
        grid_size (int): Number of points along each axis. Default: 10
        latent_dim (int): Dimension of the latent space (must be 2). Default: 2
        device (str or torch.device): Device to generate on. Default: 'cpu'
        z_range (tuple): Range for sampling (min, max). Default: (-3, 3)
    
    Returns:
        np.ndarray: Grid of generated images
    
    Example:
        >>> vae = VAE(latent_dim=2)
        >>> grid = generate_grid(vae, grid_size=10, device='cuda')
    """
    if latent_dim != 2:
        raise ValueError("Grid generation only works for 2D latent space")
    
    model.eval()
    
    # Create grid points
    z_min, z_max = z_range
    z1 = np.linspace(z_min, z_max, grid_size)
    z2 = np.linspace(z_min, z_max, grid_size)
    
    grid_points = []
    for i in z2:
        for j in z1:
            grid_points.append([j, i])
    
    grid_points = torch.tensor(grid_points, dtype=torch.float32).to(device)
    
    # Generate images
    with torch.no_grad():
        if hasattr(model, 'decoder'):
            images = model.decoder(grid_points)
        else:
            images = model(grid_points)
    
    return images.cpu().numpy()


def visualize_samples(samples, grid_shape=None, figsize=(12, 6), cmap='gray', save_path=None):
    """
    Visualize generated samples in a grid layout.
    
    Args:
        samples (torch.Tensor or np.ndarray): Generated images
        grid_shape (tuple): Grid shape (rows, cols). If None, auto-calculate. Default: None
        figsize (tuple): Figure size. Default: (12, 6)
        cmap (str): Colormap for visualization. Default: 'gray'
        save_path (str): Path to save the figure. If None, display only. Default: None
    
    Example:
        >>> samples = generate_samples(vae, num_samples=18)
        >>> visualize_samples(samples, grid_shape=(3, 6))
    """
    # Convert to numpy if needed
    if torch.is_tensor(samples):
        samples = samples.cpu().numpy()
    
    num_samples = samples.shape[0]
    
    # Auto-calculate grid shape if not provided
    if grid_shape is None:
        cols = min(6, num_samples)
        rows = (num_samples + cols - 1) // cols
        grid_shape = (rows, cols)
    
    rows, cols = grid_shape
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Handle single row/col case
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot samples
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < num_samples:
                axes[i, j].imshow(samples[idx].squeeze(), cmap=cmap)
                axes[i, j].axis('off')
            else:
                axes[i, j].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def visualize_reconstruction(model, data_loader, device='cpu', num_samples=5, save_path=None):
    """
    Visualize original images and their reconstructions side by side.
    
    Args:
        model: Trained VAE or Autoencoder model
        data_loader: DataLoader with test images
        device (str or torch.device): Device to use. Default: 'cpu'
        num_samples (int): Number of samples to visualize. Default: 5
        save_path (str): Path to save the figure. If None, display only. Default: None
    
    Example:
        >>> visualize_reconstruction(vae, test_loader, device='cuda', num_samples=5)
    """
    model.eval()
    
    # Get a batch of images
    images, _ = next(iter(data_loader))
    images = images[:num_samples].to(device)
    
    # Get reconstructions
    with torch.no_grad():
        if hasattr(model, 'forward'):
            output = model(images)
            if isinstance(output, tuple):
                reconstructions = output[0]
            else:
                reconstructions = output
        else:
            reconstructions = model(images)
    
    # Convert to numpy
    images = images.cpu().numpy()
    reconstructions = reconstructions.cpu().numpy()
    
    # Visualize
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
    
    for i in range(num_samples):
        # Original
        axes[0, i].imshow(images[i].squeeze(), cmap='gray')
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')
        
        # Reconstruction
        axes[1, i].imshow(reconstructions[i].squeeze(), cmap='gray')
        axes[1, i].set_title('Reconstructed')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved reconstruction comparison to {save_path}")
    
    plt.show()


def interpolate_latent(model, z1, z2, num_steps=10, device='cpu'):
    """
    Interpolate between two points in latent space.
    
    Args:
        model: Trained VAE model with decoder
        z1 (torch.Tensor): First latent vector
        z2 (torch.Tensor): Second latent vector
        num_steps (int): Number of interpolation steps. Default: 10
        device (str or torch.device): Device to use. Default: 'cpu'
    
    Returns:
        np.ndarray: Interpolated images
    
    Example:
        >>> z1 = torch.randn(1, 2)
        >>> z2 = torch.randn(1, 2)
        >>> interpolated = interpolate_latent(vae, z1, z2, num_steps=10)
    """
    model.eval()
    
    # Create interpolation points
    alphas = torch.linspace(0, 1, num_steps).to(device)
    z_interp = []
    
    for alpha in alphas:
        z = (1 - alpha) * z1 + alpha * z2
        z_interp.append(z)
    
    z_interp = torch.cat(z_interp, dim=0)
    
    # Generate images
    with torch.no_grad():
        if hasattr(model, 'decoder'):
            images = model.decoder(z_interp)
        else:
            images = model(z_interp)
    
    return images.cpu().numpy()


__all__ = [
    'generate_samples',
    'generate_grid',
    'visualize_samples',
    'visualize_reconstruction',
    'interpolate_latent'
]
