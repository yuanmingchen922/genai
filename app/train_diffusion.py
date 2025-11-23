"""
Training script for Diffusion Model on CIFAR-10
"""

import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from diffusion_model import get_diffusion_model, DiffusionModel


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


def train_diffusion_model(
    epochs=50,
    batch_size=128,
    learning_rate=2e-4,
    device='cpu',
    save_dir='./models',
    data_root='./data'
):
    """
    Train Diffusion Model on CIFAR-10.
    
    Args:
        epochs: number of training epochs
        batch_size: batch size for training
        learning_rate: learning rate for optimizer
        device: device to train on
        save_dir: directory to save models
        data_root: directory for CIFAR-10 data
    """
    print("=" * 70)
    print("Training Diffusion Model on CIFAR-10")
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
    print(f"\n[2] Creating Diffusion Model...")
    diffusion = get_diffusion_model(device)
    print(f"    Timesteps: {diffusion.timesteps}")
    print(f"    Device: {device}")
    
    # Count parameters
    num_params = sum(p.numel() for p in diffusion.model.parameters())
    print(f"    Model parameters: {num_params:,}")
    
    # Optimizer
    optimizer = optim.Adam(diffusion.model.parameters(), lr=learning_rate)
    
    print(f"\n[3] Training configuration:")
    print(f"    Epochs: {epochs}")
    print(f"    Learning rate: {learning_rate}")
    print(f"    Optimizer: Adam")
    
    # Training history
    history = {
        'loss': [],
        'epoch_loss': []
    }
    
    # Training loop
    print(f"\n[4] Starting training...")
    diffusion.model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch_idx, (images, _) in enumerate(progress_bar):
            images = images.to(device)
            batch_size_actual = images.size(0)
            
            # Random timesteps
            t = torch.randint(0, diffusion.timesteps, (batch_size_actual,), device=device).long()
            
            # Calculate loss
            loss = diffusion.p_losses(images, t)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Record loss
            loss_value = loss.item()
            epoch_loss += loss_value
            history['loss'].append(loss_value)
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss_value:.4f}'})
            
            # Print progress every 100 batches
            if batch_idx % 100 == 0:
                print(f"[Epoch {epoch+1}/{epochs}] [Batch {batch_idx}/{len(dataloader)}] "
                      f"[Loss: {loss_value:.4f}]")
        
        # Epoch statistics
        avg_epoch_loss = epoch_loss / len(dataloader)
        history['epoch_loss'].append(avg_epoch_loss)
        print(f"\nEpoch {epoch+1}/{epochs} completed - Average Loss: {avg_epoch_loss:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f'diffusion_cifar10_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': diffusion.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
            
            # Generate samples
            print("Generating sample images...")
            samples = diffusion.sample(16, 3, 32, 32)
            save_sample_images(samples, os.path.join(save_dir, f'diffusion_samples_epoch{epoch+1}.png'))
    
    # Save final model
    final_path = os.path.join(save_dir, 'diffusion_cifar10_final.pth')
    torch.save({
        'model_state_dict': diffusion.model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
    }, final_path)
    print(f"\n[5] Final model saved to {final_path}")
    
    # Plot training curve
    plot_training_curve(history, os.path.join(save_dir, 'diffusion_training_curve.png'))
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    
    return diffusion, history


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


def plot_training_curve(history, save_path):
    """Plot training loss curve."""
    plt.figure(figsize=(12, 5))
    
    # Batch-level loss
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], alpha=0.6)
    plt.title('Training Loss (per batch)')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    # Epoch-level loss
    plt.subplot(1, 2, 2)
    plt.plot(history['epoch_loss'], marker='o')
    plt.title('Training Loss (per epoch)')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Training curve saved to {save_path}")


if __name__ == "__main__":
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Train model
    diffusion, history = train_diffusion_model(
        epochs=50,
        batch_size=128,
        learning_rate=2e-4,
        device=device,
        save_dir='./models',
        data_root='./data'
    )
    
    # Generate final samples
    print("\nGenerating final samples...")
    samples = diffusion.sample(64, 3, 32, 32)
    save_sample_images(samples, './models/diffusion_final_samples.png')

