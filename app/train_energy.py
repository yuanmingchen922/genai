"""
Training script for Energy-Based Model on CIFAR-10
"""

import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from energy_model import get_energy_model, EnergyModel


def get_cifar10_dataloader(batch_size=128, data_root='./data'):
    """Load CIFAR-10 dataset with appropriate transforms."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    
    dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    return dataloader


def train_energy_model(
    epochs=50,
    batch_size=128,
    learning_rate=1e-4,
    device='cpu',
    save_dir='./models',
    data_root='./data'
):
    """
    Train Energy-Based Model on CIFAR-10.
    
    Args:
        epochs: number of training epochs
        batch_size: batch size for training
        learning_rate: learning rate for optimizer
        device: device to train on
        save_dir: directory to save models
        data_root: directory for CIFAR-10 data
    """
    print("=" * 70)
    print("Training Energy-Based Model on CIFAR-10")
    print("=" * 70)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load data
    print(f"\n[1] Loading CIFAR-10 dataset...")
    dataloader = get_cifar10_dataloader(batch_size, data_root)
    print(f"    Dataset size: {len(dataloader.dataset)} images")
    print(f"    Batch size: {batch_size}")
    print(f"    Batches per epoch: {len(dataloader)}")
    
    # Create model
    print(f"\n[2] Creating Energy-Based Model...")
    energy_model = get_energy_model(device, use_improved=True)
    print(f"    Langevin steps: {energy_model.langevin_steps}")
    print(f"    Langevin LR: {energy_model.langevin_lr}")
    print(f"    Langevin noise: {energy_model.langevin_noise}")
    print(f"    Device: {device}")
    
    # Count parameters
    num_params = sum(p.numel() for p in energy_model.model.parameters())
    print(f"    Model parameters: {num_params:,}")
    
    # Optimizer
    optimizer = optim.Adam(energy_model.model.parameters(), lr=learning_rate)
    
    print(f"\n[3] Training configuration:")
    print(f"    Epochs: {epochs}")
    print(f"    Learning rate: {learning_rate}")
    print(f"    Optimizer: Adam")
    
    # Training history
    history = {
        'loss': [],
        'energy_real': [],
        'energy_fake': [],
        'energy_gap': [],
        'epoch_loss': []
    }
    
    # Training loop
    print(f"\n[4] Starting training...")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_energy_real = 0.0
        epoch_energy_fake = 0.0
        epoch_energy_gap = 0.0
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch_idx, (images, _) in enumerate(progress_bar):
            images = images.to(device)
            
            # Training step
            stats = energy_model.train_step(images, optimizer)
            
            # Record statistics
            epoch_loss += stats['loss']
            epoch_energy_real += stats['energy_real']
            epoch_energy_fake += stats['energy_fake']
            epoch_energy_gap += stats['energy_gap']
            
            history['loss'].append(stats['loss'])
            history['energy_real'].append(stats['energy_real'])
            history['energy_fake'].append(stats['energy_fake'])
            history['energy_gap'].append(stats['energy_gap'])
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{stats['loss']:.4f}",
                'E_real': f"{stats['energy_real']:.2f}",
                'E_fake': f"{stats['energy_fake']:.2f}",
                'gap': f"{stats['energy_gap']:.2f}"
            })
            
            # Print progress every 100 batches
            if batch_idx % 100 == 0:
                print(f"[Epoch {epoch+1}/{epochs}] [Batch {batch_idx}/{len(dataloader)}] "
                      f"[Loss: {stats['loss']:.4f}] "
                      f"[E_real: {stats['energy_real']:.2f}] "
                      f"[E_fake: {stats['energy_fake']:.2f}] "
                      f"[Gap: {stats['energy_gap']:.2f}]")
        
        # Epoch statistics
        num_batches = len(dataloader)
        avg_loss = epoch_loss / num_batches
        avg_energy_real = epoch_energy_real / num_batches
        avg_energy_fake = epoch_energy_fake / num_batches
        avg_energy_gap = epoch_energy_gap / num_batches
        
        history['epoch_loss'].append(avg_loss)
        
        print(f"\nEpoch {epoch+1}/{epochs} completed:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Average Energy (Real): {avg_energy_real:.2f}")
        print(f"  Average Energy (Fake): {avg_energy_fake:.2f}")
        print(f"  Average Energy Gap: {avg_energy_gap:.2f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f'energy_cifar10_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': energy_model.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
            
            # Generate samples
            print("Generating sample images...")
            samples = energy_model.sample(16, 3, 32, 32)
            save_sample_images(samples, os.path.join(save_dir, f'energy_samples_epoch{epoch+1}.png'))
    
    # Save final model
    final_path = os.path.join(save_dir, 'energy_cifar10_final.pth')
    torch.save({
        'model_state_dict': energy_model.model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
    }, final_path)
    print(f"\n[5] Final model saved to {final_path}")
    
    # Plot training curves
    plot_training_curves(history, os.path.join(save_dir, 'energy_training_curves.png'))
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    
    return energy_model, history


def save_sample_images(samples, save_path):
    """Save generated samples as a grid image."""
    # Denormalize from [-1, 1] to [0, 1]
    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)
    
    # Create grid
    from torchvision.utils import make_grid
    grid = make_grid(samples, nrow=4, padding=2)
    
    # Plot
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).cpu())
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Sample images saved to {save_path}")


def plot_training_curves(history, save_path):
    """Plot training curves for energy model."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss curve
    axes[0, 0].plot(history['loss'], alpha=0.6)
    axes[0, 0].set_title('Training Loss (per batch)')
    axes[0, 0].set_xlabel('Batch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Energy curves
    axes[0, 1].plot(history['energy_real'], label='Real', alpha=0.7)
    axes[0, 1].plot(history['energy_fake'], label='Fake', alpha=0.7)
    axes[0, 1].set_title('Energy Values (per batch)')
    axes[0, 1].set_xlabel('Batch')
    axes[0, 1].set_ylabel('Energy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Energy gap
    axes[1, 0].plot(history['energy_gap'], alpha=0.7, color='purple')
    axes[1, 0].set_title('Energy Gap (Real - Fake)')
    axes[1, 0].set_xlabel('Batch')
    axes[1, 0].set_ylabel('Energy Gap')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Epoch loss
    axes[1, 1].plot(history['epoch_loss'], marker='o')
    axes[1, 1].set_title('Training Loss (per epoch)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Average Loss')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Training curves saved to {save_path}")


if __name__ == "__main__":
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Train model
    energy_model, history = train_energy_model(
        epochs=50,
        batch_size=128,
        learning_rate=1e-4,
        device=device,
        save_dir='./models',
        data_root='./data'
    )
    
    # Generate final samples
    print("\nGenerating final samples...")
    samples = energy_model.sample(64, 3, 32, 32)
    save_sample_images(samples, './models/energy_final_samples.png')

