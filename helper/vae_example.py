"""
VAE Usage Example
Demonstrates how to train and use a Variational Autoencoder for image generation.
"""

import torch
import torch.optim as optim
import numpy as np

# Import helper modules
from helper.data_loader import load_fashion_mnist, get_dataset_info
from helper.GAN_model import VAE
from helper.trainer import train_vae_model
from helper.generator import (
    generate_samples,
    visualize_samples,
    visualize_reconstruction,
    interpolate_latent
)


def preprocess_for_vae(img):
    """Pad images from 28x28 to 32x32 for VAE."""
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor()
    ])
    return transform(img)


def main():
    """Main function demonstrating VAE workflow."""
    
    print("=" * 70)
    print("VAE USAGE EXAMPLE - Fashion-MNIST Image Generation")
    print("=" * 70)
    
    # 1. Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[1] Using device: {device}")
    
    # 2. Load dataset (32x32 padded images for VAE)
    print("\n[2] Loading Fashion-MNIST dataset (padded to 32x32)...")
    
    # Create custom transform for padding
    import torchvision
    import torchvision.transforms as transforms
    
    transform = transforms.Compose([
        transforms.Pad(2),  # Pad from 28x28 to 32x32
        transforms.ToTensor()
    ])
    
    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = torchvision.datasets.FashionMNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    dataset_info = get_dataset_info('fashion_mnist')
    print(f"    Dataset: {dataset_info['name']}")
    print(f"    Classes: {dataset_info['num_classes']}")
    print(f"    Padded input shape: (1, 32, 32)")
    
    # 3. Create VAE model
    print("\n[3] Creating VAE model...")
    latent_dim = 2  # 2D latent space for easy visualization
    vae = VAE(latent_dim=latent_dim, input_channels=1)
    vae = vae.to(device)
    
    total_params = sum(p.numel() for p in vae.parameters())
    print(f"    Model: VAE with {latent_dim}D latent space")
    print(f"    Parameters: {total_params:,}")
    print(f"    Encoder: Conv layers (32->64->128)")
    print(f"    Decoder: TransposeConv layers (128->64->32)")
    
    # 4. Setup training
    print("\n[4] Setting up training...")
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    beta = 500  # Weight for reconstruction loss
    print(f"    Optimizer: Adam (lr=0.001)")
    print(f"    Loss: BCE (Reconstruction) + KL Divergence")
    print(f"    Beta (BCE weight): {beta}")
    
    # 5. Train VAE
    print("\n[5] Training VAE...")
    history = train_vae_model(
        model=vae,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        epochs=5,
        beta=beta,
        verbose=True
    )
    
    # Print training summary
    print("\n[6] Training Summary:")
    print(f"    Final total loss: {history['train_loss'][-1]:.4f}")
    print(f"    Final reconstruction loss: {history['recon_loss'][-1]:.4f}")
    print(f"    Final KL divergence: {history['kl_loss'][-1]:.4f}")
    
    # 6. Visualize reconstructions
    print("\n[7] Visualizing reconstructions...")
    visualize_reconstruction(
        model=vae,
        data_loader=test_loader,
        device=device,
        num_samples=5,
        save_path='vae_reconstructions.png'
    )
    
    # 7. Generate new samples
    print("\n[8] Generating new samples from latent space...")
    samples = generate_samples(
        model=vae,
        num_samples=18,
        latent_dim=latent_dim,
        device=device,
        return_tensor=False
    )
    
    visualize_samples(
        samples=samples,
        grid_shape=(3, 6),
        figsize=(12, 6),
        save_path='vae_generated_samples.png'
    )
    
    # 8. Latent space interpolation
    print("\n[9] Demonstrating latent space interpolation...")
    z1 = torch.randn(1, latent_dim).to(device)
    z2 = torch.randn(1, latent_dim).to(device)
    
    interpolated = interpolate_latent(
        model=vae,
        z1=z1,
        z2=z2,
        num_steps=10,
        device=device
    )
    
    visualize_samples(
        samples=interpolated,
        grid_shape=(1, 10),
        figsize=(15, 2),
        save_path='vae_interpolation.png'
    )
    
    # 9. Save model
    print("\n[10] Saving model...")
    model_path = 'vae_fashion_mnist.pth'
    torch.save(vae.state_dict(), model_path)
    print(f"    Model saved to: {model_path}")
    
    print("\n" + "=" * 70)
    print("VAE TRAINING AND GENERATION COMPLETE!")
    print("=" * 70)
    print(f"Generated images saved:")
    print(f"  - vae_reconstructions.png")
    print(f"  - vae_generated_samples.png")
    print(f"  - vae_interpolation.png")
    print("=" * 70)


def example_generate_from_saved_model():
    """Example showing how to load and generate from a saved VAE."""
    
    print("\n" + "=" * 70)
    print("GENERATING FROM SAVED VAE MODEL")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    vae = VAE(latent_dim=2, input_channels=1)
    vae.load_state_dict(torch.load('vae_fashion_mnist.pth'))
    vae.to(device)
    vae.eval()
    
    # Generate new samples
    print("\nGenerating 20 new samples...")
    samples = generate_samples(
        model=vae,
        num_samples=20,
        latent_dim=2,
        device=device
    )
    
    visualize_samples(
        samples=samples,
        grid_shape=(4, 5),
        save_path='vae_new_generation.png'
    )
    
    print("New samples saved to: vae_new_generation.png")
    print("=" * 70)


if __name__ == "__main__":
    # Run main VAE training and generation example
    main()
    
    # Uncomment to generate from saved model
    # example_generate_from_saved_model()
