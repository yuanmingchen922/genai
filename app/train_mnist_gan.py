"""
Training script for MNIST GAN
Trains a GAN model on MNIST dataset to generate hand-written digits
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
from mnist_gan_model import MNISTGenerator, MNISTDiscriminator


def train_mnist_gan(
    epochs=50,
    batch_size=128,
    learning_rate=0.0002,
    beta1=0.5,
    noise_dim=100,
    save_dir='./models',
    device=None
):
    """
    Train MNIST GAN model.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizers
        beta1: Beta1 parameter for Adam optimizer
        noise_dim: Dimension of noise vector
        save_dir: Directory to save trained models
        device: Device to train on (None for auto-detect)
    
    Returns:
        Dictionary with training history
    """
    # Setup device
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Data transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        transform=transform,
        download=True
    )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    print(f"Dataset loaded: {len(train_dataset)} images")
    print(f"Number of batches: {len(train_loader)}")
    
    # Initialize models
    print("\nInitializing models...")
    generator = MNISTGenerator(noise_dim=noise_dim).to(device)
    discriminator = MNISTDiscriminator().to(device)
    
    # Print model info
    gen_params = sum(p.numel() for p in generator.parameters())
    disc_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Generator parameters: {gen_params:,}")
    print(f"Discriminator parameters: {disc_params:,}")
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    
    # Labels
    real_label = 1.0
    fake_label = 0.0
    
    # Fixed noise for visualization
    fixed_noise = torch.randn(64, noise_dim).to(device)
    
    # Training history
    g_losses = []
    d_losses = []
    
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)
    
    # Training loop
    for epoch in range(epochs):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for i, (real_images, _) in enumerate(pbar):
            batch_size_actual = real_images.size(0)
            real_images = real_images.to(device)
            
            # Create labels
            real_labels = torch.full((batch_size_actual, 1), real_label, dtype=torch.float, device=device)
            fake_labels = torch.full((batch_size_actual, 1), fake_label, dtype=torch.float, device=device)
            
            # ========================================
            # Train Discriminator
            # ========================================
            optimizer_d.zero_grad()
            
            # Train on real images
            output_real = discriminator(real_images)
            loss_d_real = criterion(output_real, real_labels)
            
            # Train on fake images
            noise = torch.randn(batch_size_actual, noise_dim).to(device)
            fake_images = generator(noise)
            output_fake = discriminator(fake_images.detach())
            loss_d_fake = criterion(output_fake, fake_labels)
            
            # Total discriminator loss
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            optimizer_d.step()
            
            # ========================================
            # Train Generator
            # ========================================
            optimizer_g.zero_grad()
            
            # Generate fake images and get discriminator's opinion
            output_fake_for_g = discriminator(fake_images)
            loss_g = criterion(output_fake_for_g, real_labels)
            
            loss_g.backward()
            optimizer_g.step()
            
            # Track losses
            epoch_g_loss += loss_g.item()
            epoch_d_loss += loss_d.item()
            
            # Update progress bar
            pbar.set_postfix({
                'D_loss': f'{loss_d.item():.4f}',
                'G_loss': f'{loss_g.item():.4f}'
            })
        
        # Average losses for the epoch
        avg_g_loss = epoch_g_loss / len(train_loader)
        avg_d_loss = epoch_d_loss / len(train_loader)
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)
        
        print(f"Epoch [{epoch+1}/{epochs}] - D_loss: {avg_d_loss:.4f}, G_loss: {avg_g_loss:.4f}")
        
        # Generate and save sample images every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            generator.eval()
            with torch.no_grad():
                fake_samples = generator(fixed_noise).cpu()
                fake_samples = fake_samples * 0.5 + 0.5  # Denormalize
                
                # Save sample images
                fig, axes = plt.subplots(4, 8, figsize=(12, 6))
                for idx, ax in enumerate(axes.flat):
                    if idx < len(fake_samples):
                        ax.imshow(fake_samples[idx].squeeze(), cmap='gray')
                    ax.axis('off')
                plt.suptitle(f"Generated Samples - Epoch {epoch+1}")
                plt.tight_layout()
                plt.savefig(f'{save_dir}/samples_epoch_{epoch+1}.png')
                plt.close()
            generator.train()
    
    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70)
    
    # Save final models
    print("\nSaving models...")
    
    torch.save({
        'epoch': epochs,
        'model_state_dict': generator.state_dict(),
        'optimizer_state_dict': optimizer_g.state_dict(),
        'g_losses': g_losses,
    }, os.path.join(save_dir, 'generator_mnist_gan.pth'))
    
    torch.save({
        'epoch': epochs,
        'model_state_dict': discriminator.state_dict(),
        'optimizer_state_dict': optimizer_d.state_dict(),
        'd_losses': d_losses,
    }, os.path.join(save_dir, 'discriminator_mnist_gan.pth'))
    
    print(f"Models saved to {save_dir}/")
    
    # Plot and save training curves
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('MNIST GAN Training Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()
    
    print(f"Training curves saved to {save_dir}/training_curves.png")
    
    # Return training history
    return {
        'epochs_completed': epochs,
        'generator_losses': g_losses,
        'discriminator_losses': d_losses,
        'final_g_loss': g_losses[-1],
        'final_d_loss': d_losses[-1],
        'device': str(device)
    }


def main():
    """Main training function."""
    print("=" * 70)
    print("MNIST GAN Training")
    print("=" * 70)
    
    # Training configuration
    config = {
        'epochs': 50,
        'batch_size': 128,
        'learning_rate': 0.0002,
        'beta1': 0.5,
        'noise_dim': 100,
        'save_dir': './models'
    }
    
    print("\nTraining Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Train the model
    history = train_mnist_gan(**config)
    
    # Print summary
    print("\n" + "=" * 70)
    print("Training Summary:")
    print("=" * 70)
    print(f"Epochs completed: {history['epochs_completed']}")
    print(f"Final generator loss: {history['final_g_loss']:.4f}")
    print(f"Final discriminator loss: {history['final_d_loss']:.4f}")
    print(f"Device used: {history['device']}")
    print("\nModels saved successfully!")
    print("You can now use the trained generator in the API.")


if __name__ == "__main__":
    main()
