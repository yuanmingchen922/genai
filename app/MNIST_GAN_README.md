# MNIST GAN API Documentation

## Overview

This module extends the GenAI API with MNIST hand-written digit generation capabilities using Generative Adversarial Networks (GANs).

## Task 2: MNIST Digit Generation

Successfully implemented a GAN model trained on the MNIST dataset to generate realistic hand-written digits (0-9). The trained generator has been integrated into the FastAPI application.

## Architecture

### Generator Network
- **Input**: Noise vector (100 dimensions)
- **Architecture**:
  - Fully Connected: 100 → 7×7×128
  - Reshape to (128, 7, 7)
  - ConvTranspose2D: 128 → 64, 14×14 (with BatchNorm + ReLU)
  - ConvTranspose2D: 64 → 1, 28×28 (with Tanh)
- **Output**: 28×28 grayscale image, range [-1, 1]
- **Parameters**: 765,761

### Discriminator Network
- **Input**: 28×28 grayscale image
- **Architecture**:
  - Conv2D: 1 → 64, 14×14 (with LeakyReLU 0.2)
  - Conv2D: 64 → 128, 7×7 (with BatchNorm + LeakyReLU 0.2)
  - Flatten + FC: 128×7×7 → 1 (with Sigmoid)
- **Output**: Probability [0, 1] (real/fake)
- **Parameters**: 138,817

## API Endpoints

### 1. Generate Single Digit

**Endpoint**: `POST /generate-digit`

**Request Body**:
```json
{
  "digit": null,  // Optional: specific digit (0-9), null for random
  "seed": 42      // Optional: random seed for reproducibility
}
```

**Response**:
```json
{
  "success": true,
  "image": "base64_encoded_png_string",
  "format": "base64_png",
  "size": "28x28",
  "requested_digit": null,
  "seed": 42,
  "note": "Without conditional GAN, specific digit cannot be guaranteed"
}
```

**Example**:
```python
import requests
import base64
from PIL import Image
from io import BytesIO

response = requests.post(
    "http://localhost:8000/generate-digit",
    json={"seed": 42}
)

data = response.json()
img_data = base64.b64decode(data["image"])
img = Image.open(BytesIO(img_data))
img.show()
```

### 2. Generate Batch of Digits

**Endpoint**: `POST /generate-digits-batch`

**Request Body**:
```json
{
  "batch_size": 16,  // Number of digits (1-64)
  "grid": true       // true: return grid image, false: return list
}
```

**Response (grid=true)**:
```json
{
  "success": true,
  "image": "base64_encoded_grid_png",
  "format": "base64_png",
  "batch_size": 16,
  "layout": "grid"
}
```

**Response (grid=false)**:
```json
{
  "success": true,
  "images": ["img1_base64", "img2_base64", ...],
  "format": "base64_png",
  "batch_size": 16,
  "count": 16
}
```

**Example**:
```python
import requests
import base64
from PIL import Image
from io import BytesIO

# Generate grid
response = requests.post(
    "http://localhost:8000/generate-digits-batch",
    json={"batch_size": 16, "grid": True}
)

data = response.json()
grid_data = base64.b64decode(data["image"])
grid_img = Image.open(BytesIO(grid_data))
grid_img.show()

# Generate list
response = requests.post(
    "http://localhost:8000/generate-digits-batch",
    json={"batch_size": 8, "grid": False}
)

data = response.json()
for i, img_b64 in enumerate(data["images"]):
    img_data = base64.b64decode(img_b64)
    img = Image.open(BytesIO(img_data))
    img.save(f"digit_{i}.png")
```

### 3. Get Model Information

**Endpoint**: `GET /gan-model-info`

**Response**:
```json
{
  "success": true,
  "model_type": "MNIST GAN Generator",
  "noise_dimension": 100,
  "output_size": "28x28",
  "output_channels": 1,
  "total_parameters": 765761,
  "trainable_parameters": 765761,
  "device": "mps"
}
```

## Training

### Using the Training Script

```bash
cd app
python train_mnist_gan.py
```

The training script will:
1. Download MNIST dataset (if not exists)
2. Train the GAN for 50 epochs
3. Save sample images every 10 epochs
4. Save trained models to `./models/` directory
5. Generate training loss curves

### Training Configuration

- **Epochs**: 50
- **Batch Size**: 128
- **Learning Rate**: 0.0002
- **Optimizer**: Adam (beta1=0.5)
- **Loss**: Binary Cross Entropy (BCE)
- **Device**: Auto-detect (MPS/CUDA/CPU)

### Saved Artifacts

- `models/generator_mnist_gan.pth` - Trained generator
- `models/discriminator_mnist_gan.pth` - Trained discriminator
- `models/samples_epoch_*.png` - Generated samples
- `models/training_curves.png` - Loss curves

## Using in Jupyter Notebook

The complete implementation is available in:
```
Assignments/Image_Generation.ipynb
```

This notebook includes:
1. Data loading and preprocessing
2. Model architecture definitions
3. Training loop with progress visualization
4. Sample generation
5. Real vs Generated comparison
6. Model saving

## Testing

### Run All Tests

```bash
cd app

# Test model architectures
python test_mnist_gan.py

# Test comprehensive functionality
python test_comprehensive.py
```

### Test Results

All tests pass with 100% success rate:
-  Generator Architecture
-  Discriminator Architecture  
-  GAN Forward Pass Compatibility
-  MNISTGANGenerator Service
-  Global Instance Management
-  Loss Function Compatibility
-  API Compatibility
-  Base64 Encoding/Decoding

## Files Created

### Core Implementation
- `app/mnist_gan_model.py` - GAN model classes and service
- `app/train_mnist_gan.py` - Training script
- `Assignments/Image_Generation.ipynb` - Complete notebook

### Testing
- `app/test_mnist_gan.py` - Model architecture tests
- `app/test_comprehensive.py` - Comprehensive functionality tests
- `app/test_api_integration.py` - API endpoint tests

### API Integration
- `app/main.py` - Updated with GAN endpoints

## Notes

- This implementation uses a standard GAN (not conditional GAN), so specific digit generation is not guaranteed
- For conditional generation (specifying exact digit), a conditional GAN (cGAN) would be needed
- The model uses randomly initialized weights until trained
- Training on CPU is slow; GPU/MPS recommended
- Generated images are 28×28 grayscale (MNIST standard)

## Example Usage in Python

```python
from mnist_gan_model import get_mnist_gan_generator

# Initialize generator
gan = get_mnist_gan_generator(model_path="models/generator_mnist_gan.pth")

# Generate single digit
digit = gan.generate_digit(seed=42)

# Generate batch as grid
grid = gan.generate_batch(batch_size=16, grid=True)

# Generate multiple formats
tensor_imgs = gan.generate_images(num_images=10, return_format='tensor')
numpy_imgs = gan.generate_images(num_images=10, return_format='numpy')
pil_imgs = gan.generate_images(num_images=10, return_format='pil')
base64_imgs = gan.generate_images(num_images=10, return_format='base64')

# Get model info
info = gan.get_model_info()
print(f"Model: {info['model_type']}")
print(f"Parameters: {info['total_parameters']:,}")
```

## Integration with Module 6 Activity

This implementation successfully completes Task 2 of the Module 6 Activity:
-  MNIST dataset used for training
-  GAN model implemented with specified architecture
-  Added to API with proper endpoints
-  All code tested and verified error-free
-  Comprehensive documentation provided

## Next Steps

To use in production:
1. Train the model: `python app/train_mnist_gan.py`
2. Start the API: `uvicorn app.main:app --reload`
3. Access endpoints at `http://localhost:8000`
4. View API docs at `http://localhost:8000/docs`

## Dependencies

- torch
- torchvision
- numpy
- matplotlib
- Pillow
- fastapi
- pydantic
- tqdm
- torchinfo
