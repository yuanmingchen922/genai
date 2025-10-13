"""
GAN Usage Example
Demonstrates how to train and use a Generative Adversarial Network for image generation.
"""

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Import helper modules
from helper.data_loader import load_fashion_mnist, CustomImageDataset, create_gan_transforms
from helper.GAN_model import Generator, Critic, GAN
from helper.trainer import train_gan
from helper.generator import (
    generate_gan_samples,
    visualize_gan_progress,
    compare_real_fake,
    interpolate_gan_latent,
    visualize_samples
)


def main():
    """Main function demonstrating GAN training workflow."""
    
    print("=" * 70)
    print("GAN USAGE EXAMPLE - Image Generation with WGAN")
    print("=" * 70)
    
    # 1. Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[1] Using device: {device}")
    
    # 2. Load dataset
    print("\n[2] Loading dataset...")
    
    # Option A: Use Fashion-MNIST (for quick testing)
    print("    Using Fashion-MNIST dataset (padded to 64x64)")
    
    import torchvision
    import torchvision.transforms as transforms
    
    # GAN-specific transforms for Fashion-MNIST
    transform = transforms.Compose([
        transforms.Pad(18),  # Pad from 28x28 to 64x64
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])
    
    # Convert grayscale to RGB for consistency
    class ToRGB:
        def __call__(self, x):
            return x.repeat(3, 1, 1) if x.size(0) == 1 else x
    
    transform = transforms.Compose([
        transform,
        ToRGB()
    ])
    
    dataset = torchvision.datasets.FashionMNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=128, 
        shuffle=True, 
        num_workers=2
    )
    
    print(f"    Dataset size: {len(dataset)} images")
    print(f"    Batch size: 128")
    
    # Option B: Use custom image dataset (uncomment to use)
    # print("    Using custom image dataset")
    # transform = create_gan_transforms(image_size=64, normalize_range='tanh')
    # dataloader = load_celeba('../data/img_align_celeba', batch_size=128, image_size=64)
    
    # 3. Create GAN models
    print("\n[3] Creating GAN models...")
    z_dim = 100
    
    generator = Generator(z_dim=z_dim, output_channels=3, image_size=64).to(device)
    critic = Critic(input_channels=3, image_size=64).to(device)
    
    # Count parameters
    gen_params = sum(p.numel() for p in generator.parameters())
    critic_params = sum(p.numel() for p in critic.parameters())
    
    print(f"    Generator parameters: {gen_params:,}")
    print(f"    Critic parameters: {critic_params:,}")
    print(f"    Noise dimension: {z_dim}")
    print(f"    Image size: 64x64x3")
    
    # 4. Setup training
    print("\n[4] Setting up training...")
    lr = 5e-5
    n_critic = 5
    clip_value = 0.01
    epochs = 4
    
    opt_gen = optim.RMSprop(generator.parameters(), lr=lr)
    opt_critic = optim.RMSprop(critic.parameters(), lr=lr)
    
    print(f"    Optimizer: RMSprop (lr={lr})")
    print(f"    Training strategy: WGAN with weight clipping")
    print(f"    Critic updates per generator update: {n_critic}")
    print(f"    Weight clip value: {clip_value}")
    print(f"    Epochs: {epochs}")
    
    # Fixed noise for visualization
    fixed_noise = torch.randn(64, z_dim).to(device)
    
    # 5. Train GAN
    print("\n[5] Training GAN...")
    history = train_gan(
        generator=generator,
        critic=critic,
        train_loader=dataloader,
        opt_gen=opt_gen,
        opt_critic=opt_critic,
        device=device,
        epochs=epochs,
        n_critic=n_critic,
        clip_value=clip_value,
        z_dim=z_dim,
        verbose=True
    )
    
    # Print training summary
    print("\n[6] Training Summary:")
    print(f"    Epochs completed: {history['epochs_completed']}")
    print(f"    Final generator loss: {history['gen_loss'][-1]:.4f}")
    print(f"    Final critic loss: {history['critic_loss'][-1]:.4f}")
    print(f"    Total batches trained: {len(history['gen_loss'])}")
    
    # 6. Generate samples
    print("\n[7] Generating samples...")
    
    # Generate new samples
    samples = generate_gan_samples(
        generator=generator,
        num_samples=64,
        z_dim=z_dim,
        device=device,
        return_tensor=True
    )
    
    # Visualize generated samples
    visualize_gan_progress(
        generator=generator,
        fixed_noise=fixed_noise,
        epoch=history['epochs_completed'],
        device=device,
        save_path='gan_final_samples.png'
    )
    
    # 7. Compare real vs fake
    print("\n[8] Comparing real vs generated images...")
    real_batch, _ = next(iter(dataloader))
    fake_batch = generate_gan_samples(generator, 8, z_dim, device, return_tensor=True)
    
    compare_real_fake(
        real_images=real_batch,
        fake_images=fake_batch,
        num_samples=8,
        save_path='gan_real_vs_fake.png'
    )
    
    # 8. Latent space interpolation
    print("\n[9] Demonstrating latent space interpolation...")
    z1 = torch.randn(1, z_dim).to(device)
    z2 = torch.randn(1, z_dim).to(device)
    
    interpolated = interpolate_gan_latent(
        generator=generator,
        z1=z1,
        z2=z2,
        num_steps=10,
        device=device
    )
    
    # Denormalize and visualize interpolation
    interpolated_denorm = (interpolated + 1) / 2  # [-1,1] -> [0,1]
    visualize_samples(
        samples=interpolated_denorm,
        grid_shape=(1, 10),
        figsize=(20, 2),
        save_path='gan_interpolation.png'
    )
    
    # 9. Save models
    print("\n[10] Saving models...")
    torch.save(generator.state_dict(), 'gan_generator.pth')
    torch.save(critic.state_dict(), 'gan_critic.pth')
    print("    Generator saved to: gan_generator.pth")
    print("    Critic saved to: gan_critic.pth")
    
    print("\n" + "=" * 70)
    print("GAN TRAINING AND GENERATION COMPLETE!")
    print("=" * 70)
    print("Generated files:")
    print("  - gan_final_samples.png: Final generated samples")
    print("  - gan_real_vs_fake.png: Real vs generated comparison")
    print("  - gan_interpolation.png: Latent space interpolation")
    print("  - gan_generator.pth: Trained generator model")
    print("  - gan_critic.pth: Trained critic model")
    print("=" * 70)


def example_generate_from_saved_model():
    """Example showing how to load and generate from a saved GAN."""
    
    print("\n" + "=" * 70)
    print("GENERATING FROM SAVED GAN MODEL")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    z_dim = 100
    
    # Load generator
    generator = Generator(z_dim=z_dim, output_channels=3, image_size=64)
    generator.load_state_dict(torch.load('gan_generator.pth'))
    generator.to(device)
    generator.eval()
    
    # Generate new samples
    print("\nGenerating 64 new samples...")
    samples = generate_gan_samples(
        generator=generator,
        num_samples=64,
        z_dim=z_dim,
        device=device
    )
    
    # Denormalize and visualize
    samples_denorm = (samples + 1) / 2  # [-1,1] -> [0,1]
    visualize_samples(
        samples=samples_denorm,
        grid_shape=(8, 8),
        figsize=(12, 12),
        save_path='gan_new_generation.png'
    )
    
    print("New samples saved to: gan_new_generation.png")
    print("=" * 70)


def plot_training_losses(history):
    """Plot training losses over time."""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['gen_loss'], label='Generator Loss', alpha=0.7)
    plt.title('Generator Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['critic_loss'], label='Critic Loss', alpha=0.7)
    plt.title('Critic Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gan_training_losses.png', bbox_inches='tight', dpi=150)
    plt.show()


if __name__ == "__main__":
    # Run main GAN training example
    main()
    
    # Uncomment to generate from saved model
    # example_generate_from_saved_model()