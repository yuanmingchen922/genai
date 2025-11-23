# Helper Library - Reusable Neural Network Components

Modular implementations for GAN, VAE, and common utilities.

## Components

- **GAN_model/** - Generator and Critic architectures
- **generator/** - Image generation utilities
- **trainer/** - Training loops
- **data_loader/** - Dataset loaders
- **evaluator/** - Model evaluation

## Usage

```python
from helper.GAN_model import Generator, Critic
from helper.trainer import train_gan
from helper.generator import generate_gan_samples

# Create models
generator = Generator(z_dim=100)
critic = Critic()

# Train
history = train_gan(generator, critic, dataloader, ...)

# Generate
samples = generate_gan_samples(generator, num_samples=64)
```

See `gan_example.py` and `vae_example.py` for complete examples.
