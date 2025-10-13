# Generative Adversarial Network (GAN) Module

This module extends the helper library with GAN capabilities for high-quality image generation using the Wasserstein GAN (WGAN) approach.

## What is GAN?

Generative Adversarial Networks (GANs) are a class of generative models that learn to create realistic data through adversarial training between two networks:

1. **Generator**: Creates fake data from random noise
2. **Discriminator/Critic**: Distinguishes between real and fake data

The two networks compete in a minimax game, where the generator aims to fool the discriminator, while the discriminator tries to correctly identify fake data.

## WGAN Implementation

This implementation uses **Wasserstein GAN (WGAN)**, which offers several advantages over vanilla GAN:

- More stable training
- Better convergence properties  
- Meaningful loss curves
- Reduced mode collapse

### Key WGAN Features:
- **Critic instead of Discriminator**: Outputs a scalar score rather than probability
- **Wasserstein Distance**: More stable than JS divergence
- **Weight Clipping**: Enforces Lipschitz constraint
- **No Sigmoid**: Critic outputs unbounded scores

## Architecture

### Generator Network
```python
Input: 100D noise vector -> Output: 64x64x3 RGB image

z (100,) -> reshape -> (100,1,1)
-> TransposeConv 100->512 (4x4)
-> TransposeConv 512->256 (8x8)  
-> TransposeConv 256->128 (16x16)
-> TransposeConv 128->64 (32x32)
-> TransposeConv 64->3 (64x64)
-> Tanh activation (output range [-1,1])
```

**Key Design Elements**:
- Transpose convolutions for upsampling
- BatchNorm for training stability
- ReLU activation (except final layer)
- Tanh output to match normalized input range

### Critic Network  
```python
Input: 64x64x3 RGB image -> Output: Scalar score

(3,64,64) -> Conv 3->64 (32x32)
-> Conv 64->128 (16x16)
-> Conv 128->256 (8x8)
-> Conv 256->512 (4x4)  
-> Conv 512->1 (1x1)
-> Flatten -> scalar
```

**Key Design Elements**:
- Convolutional layers for downsampling
- LeakyReLU(0.2) activation
- BatchNorm (except first layer)
- No final activation function

## Usage Example

### Basic Training
```python
from helper.model import Generator, Critic
from helper.trainer import train_gan
from helper.data_loader import load_celeba

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
z_dim = 100

# Create models
generator = Generator(z_dim=z_dim).to(device)
critic = Critic().to(device)

# Setup optimizers
opt_gen = torch.optim.RMSprop(generator.parameters(), lr=5e-5)
opt_critic = torch.optim.RMSprop(critic.parameters(), lr=5e-5)

# Load data
dataloader = load_celeba('../data/img_align_celeba', batch_size=128)

# Train
history = train_gan(
    generator=generator,
    critic=critic, 
    train_loader=dataloader,
    opt_gen=opt_gen,
    opt_critic=opt_critic,
    device=device,
    epochs=10
)
```

### Generation
```python
from helper.generator import generate_gan_samples, visualize_gan_progress

# Generate samples
samples = generate_gan_samples(generator, num_samples=64, z_dim=100, device=device)

# Visualize progress
fixed_noise = torch.randn(64, 100).to(device)
visualize_gan_progress(generator, fixed_noise, epoch=10, device=device)
```

## Complete Workflow

Run the complete example:
```bash
python helper/gan_example.py
```

This demonstrates:
1. **Data Loading**: Fashion-MNIST with GAN-appropriate preprocessing
2. **Model Creation**: Generator and Critic networks
3. **WGAN Training**: 4 epochs with progress monitoring
4. **Sample Generation**: Create new images from noise
5. **Visualization**: Compare real vs fake images
6. **Interpolation**: Smooth transitions in latent space
7. **Model Saving**: Save trained weights

## Key Components

### Model Module (`helper/model/__init__.py`)
- `Generator`: Deep convolutional generator with transpose convolutions
- `Critic`: Convolutional critic for image evaluation  
- `GAN`: Combined wrapper class

### Trainer Module (`helper/trainer/__init__.py`)  
- `train_gan()`: Complete WGAN training loop with:
  - Critic training (5x per generator update)
  - Weight clipping for Lipschitz constraint
  - Progress monitoring and loss tracking
- `generate_gan_samples()`: Quick sample generation

### Data Loader Module (`helper/data_loader/__init__.py`)
- `CustomImageDataset`: Load images from directory
- `load_celeba()`: CelebA dataset with GAN preprocessing
- `create_gan_transforms()`: Standard GAN image transforms

### Generator Module (`helper/generator/__init__.py`)
- `generate_gan_samples()`: Generate images from trained GAN
- `visualize_gan_progress()`: Monitor training progress
- `compare_real_fake()`: Side-by-side comparison
- `interpolate_gan_latent()`: Latent space interpolation

## Training Parameters

### Core Parameters
- `z_dim = 100`: Noise vector dimension
- `lr = 5e-5`: Learning rate (smaller than typical)
- `batch_size = 128`: Training batch size
- `epochs = 10`: Number of training epochs

### WGAN-Specific
- `n_critic = 5`: Critic updates per generator update
- `clip_value = 0.01`: Weight clipping range [-0.01, 0.01]
- `optimizer = RMSprop`: Better than Adam for WGAN

### Data Preprocessing
```python
transforms.Compose([
    transforms.CenterCrop(178),      # Remove background
    transforms.Resize(64),           # Standard size
    transforms.ToTensor(),           # Convert to tensor
    transforms.Normalize([0.5]*3, [0.5]*3)  # Normalize to [-1,1]
])
```

## Loss Function

WGAN uses Wasserstein distance instead of cross-entropy:

```python
# Critic loss: maximize real scores, minimize fake scores
loss_critic = -(critic(real).mean() - critic(fake).mean())

# Generator loss: maximize critic scores for fake images  
loss_gen = -critic(fake).mean()
```

## Advanced Features

### Latent Space Interpolation
```python
z1 = torch.randn(1, 100)
z2 = torch.randn(1, 100)
interpolated = interpolate_gan_latent(generator, z1, z2, num_steps=10)
```

### Custom Dataset Training
```python
from helper.data_loader import CustomImageDataset, create_gan_transforms

transform = create_gan_transforms(image_size=128, normalize_range='tanh')
dataset = CustomImageDataset('../data/my_images', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
```

### Progress Visualization
```python
# Monitor training with fixed noise
fixed_noise = torch.randn(64, 100).to(device)

# After each epoch
visualize_gan_progress(generator, fixed_noise, epoch, save_path=f'epoch_{epoch}.png')
```

## Output Files

Running the example generates:
- `gan_generator.pth`: Trained generator weights
- `gan_critic.pth`: Trained critic weights  
- `gan_final_samples.png`: Final generated samples
- `gan_real_vs_fake.png`: Real vs generated comparison
- `gan_interpolation.png`: Latent space interpolation
- `gan_training_losses.png`: Loss curves

## Tips for Success

### Training Stability
1. **Use RMSprop optimizer** with lr=5e-5
2. **Train critic 5x more** than generator
3. **Clip weights** to [-0.01, 0.01] 
4. **Normalize images** to [-1, 1] range
5. **Use BatchNorm** in both networks

### Data Preparation
1. **Center crop** to remove unnecessary background
2. **Resize** to power-of-2 dimensions (64x64, 128x128)
3. **Normalize** to [-1, 1] for tanh output
4. **Augment** data if dataset is small

### Monitoring Training
1. **Watch loss curves** (should be relatively stable)
2. **Generate samples** with fixed noise each epoch
3. **Compare real vs fake** images regularly
4. **Check for mode collapse** (diversity in outputs)

## Common Issues & Solutions

### Mode Collapse
- **Symptoms**: Generator produces limited variety
- **Solutions**: Reduce learning rate, increase critic training, try different architectures

### Training Instability  
- **Symptoms**: Loss oscillations, poor sample quality
- **Solutions**: Lower learning rates, proper weight initialization, gradient penalties

### Poor Image Quality
- **Symptoms**: Blurry or unrealistic images
- **Solutions**: More training epochs, better architecture, larger dataset

## Extensions

This GAN implementation can be extended with:
- **Progressive Growing**: Start with low resolution, gradually increase
- **Self-Attention**: Add attention mechanisms for better details
- **Spectral Normalization**: Alternative to weight clipping
- **Conditional Generation**: Control generation with class labels
- **Style Transfer**: Modify existing images instead of generating new ones

## References

Based on Module 6 materials:
- Wasserstein GAN theory and implementation
- Deep Convolutional GAN (DCGAN) architecture
- CelebA dataset preprocessing
- Adversarial training strategies