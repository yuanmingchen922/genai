# Variational Autoencoder (VAE) Module

This module extends the helper library with Variational Autoencoder capabilities for generative modeling.

## What is VAE?

Variational Autoencoder (VAE) is a generative model that learns to:
1. Encode images into a continuous latent space (distribution)
2. Sample from this latent space
3. Decode samples back into realistic images

Unlike regular autoencoders, VAE enforces structure in the latent space by:
- Mapping to distributions (mean and variance) instead of fixed points
- Using the reparameterization trick for backpropagation
- Adding KL divergence loss to regularize the latent space

## Architecture

### VariationalEncoder
- Input: 32x32 grayscale image
- Conv layers: 1 → 32 → 64 → 128 channels
- Output: Two vectors (mean and log variance) of latent_dim

### VariationalDecoder
- Input: Latent vector (e.g., 2D)
- FC layer to expand to 128 * 4 * 4
- TransposeConv layers: 128 → 64 → 32 → 1 channels
- Output: 32x32 reconstructed image

### VAE
- Combines encoder and decoder
- Implements reparameterization trick: z = μ + σ * ε
- Loss = β * BCE(reconstruction) + KL(distribution || N(0,1))

## Usage Example

```python
import torch
from helper.model import VAE
from helper.trainer import train_vae_model
from helper.generator import generate_samples, visualize_samples

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create VAE with 2D latent space
vae = VAE(latent_dim=2, input_channels=1).to(device)

# Train
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
history = train_vae_model(vae, train_loader, optimizer, device, epochs=10, beta=500)

# Generate new samples
samples = generate_samples(vae, num_samples=20, latent_dim=2, device=device)
visualize_samples(samples, grid_shape=(4, 5))
```

## Run Complete Example

```bash
python helper/vae_example.py
```

This will:
1. Load Fashion-MNIST dataset (padded to 32x32)
2. Train a VAE model
3. Visualize reconstructions
4. Generate new samples from latent space
5. Show latent space interpolation
6. Save model and generated images

## Key Features

### Model Module
- `VariationalEncoder`: Encoder with mean and log variance outputs
- `VariationalDecoder`: Decoder with transpose convolutions
- `VAE`: Complete VAE with reparameterization trick
- `VAE.generate()`: Generate samples directly from the model

### Trainer Module
- `train_vae_model()`: Train VAE with reconstruction + KL loss
- `vae_loss_function()`: Combined BCE and KL divergence loss
- Tracks reconstruction loss and KL divergence separately

### Generator Module
- `generate_samples()`: Sample from standard normal and decode
- `generate_grid()`: Generate grid of samples (for 2D latent space)
- `visualize_samples()`: Display generated images in grid
- `visualize_reconstruction()`: Compare original and reconstructed images
- `interpolate_latent()`: Smooth interpolation between two latent points

## Loss Function

The VAE loss has two components:

1. Reconstruction Loss (BCE): Measures how well the decoder reconstructs the input
   ```
   BCE = -Σ[x * log(x') + (1-x) * log(1-x')]
   ```

2. KL Divergence: Regularizes the latent space to be close to N(0,1)
   ```
   KLD = -0.5 * Σ[1 + log(σ²) - μ² - σ²]
   ```

Total loss:
```
Loss = β * BCE + KLD
```

where β (default 500) controls the trade-off between reconstruction quality and latent space structure.

## Reparameterization Trick

To enable backpropagation through sampling:

```python
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)  # Random noise from N(0,1)
    return mu + eps * std        # Sample from N(mu, std²)
```

This separates the stochastic and deterministic parts, allowing gradients to flow through μ and σ.

## Data Preprocessing

Fashion-MNIST images are padded from 28x28 to 32x32:

```python
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Pad(2),      # Pad to 32x32
    transforms.ToTensor()   # Convert to tensor
])
```

This makes it easier to design symmetric encoder-decoder architectures with stride-2 convolutions.

## Latent Space Visualization

For 2D latent space, you can visualize the learned structure:

```python
from helper.generator import generate_grid, visualize_samples

# Generate grid spanning the latent space
grid = generate_grid(vae, grid_size=10, z_range=(-3, 3), device=device)
visualize_samples(grid, grid_shape=(10, 10))
```

## Interpolation

Generate smooth transitions between two images:

```python
from helper.generator import interpolate_latent

# Encode two images to get their latent representations
mu1, _ = vae.encoder(image1)
mu2, _ = vae.encoder(image2)

# Interpolate
interpolated = interpolate_latent(vae, mu1, mu2, num_steps=10)
visualize_samples(interpolated, grid_shape=(1, 10))
```

## Parameters

### Model Parameters
- `latent_dim`: Dimension of latent space (default: 2)
  - Small values (2-3): Good for visualization
  - Large values (32-128): Better generation quality
- `input_channels`: Number of input channels (1 for grayscale)

### Training Parameters
- `epochs`: Number of training epochs (default: 10)
- `lr`: Learning rate (default: 1e-3)
- `beta`: Weight for reconstruction loss (default: 500)
  - Higher β: Better reconstruction, less structured latent space
  - Lower β: More structured latent space, potentially worse reconstruction

## Output Files

Running the example generates:
- `vae_fashion_mnist.pth`: Trained model weights
- `vae_reconstructions.png`: Original vs reconstructed images
- `vae_generated_samples.png`: New samples from latent space
- `vae_interpolation.png`: Interpolation between two points

## Advanced Usage

### Custom Latent Dimension

```python
vae = VAE(latent_dim=32)  # Higher dimensional latent space
```

### Custom Beta Schedule

```python
# Start with low beta, increase gradually
for epoch in range(epochs):
    beta = min(500, epoch * 50)
    train_vae_model(vae, train_loader, optimizer, device, epochs=1, beta=beta)
```

### Generate from Specific Region

```python
# Generate samples near a specific point in latent space
z_center = torch.tensor([[1.0, 2.0]]).to(device)
z_noise = torch.randn(10, 2).to(device) * 0.5  # Small noise
z = z_center + z_noise

with torch.no_grad():
    samples = vae.decoder(z)
```

## References

Based on Module 5 from the course materials:
- Variational Autoencoders theory
- Reparameterization trick
- Fashion-MNIST dataset preparation
- Latent space visualization techniques
