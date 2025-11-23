# Run Instructions - GenAI Project

## Quick Start for Anyone

This guide ensures anyone can run the project successfully without errors.

## Prerequisites

- Python 3.10 or higher
- pip package manager
- 4GB+ RAM
- Internet connection (for first-time setup)

## Step-by-Step Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yuanmingchen922/genai.git
cd genai
```

### 2. Install Dependencies

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt

# Install spaCy model (if not auto-installed)
pip install en_core_web_lg-3.7.1-py3-none-any.whl
```

### 3. Verify Installation

```bash
python verify_all_models.py
```

You should see:
```
✅ ALL VERIFICATIONS PASSED!
Results: 5/5 tests passed
```

### 4. Run the Application

#### Option A: Local API Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Then visit: http://localhost:8000/docs

#### Option B: Docker

```bash
# Validate Docker setup
./validate_docker_setup.sh

# Build and run
./docker-build.sh

# Or manually
docker-compose up --build -d
```

#### Option C: Jupyter Notebook

```bash
cd Assignments
jupyter notebook Advanced_Image_Generation_my2878.ipynb
```

### 5. Test the API

```bash
# Test all endpoints
python app/test_assignment4_api.py

# Quick health check
curl http://localhost:8000/health
```

## Testing Individual Models

### Test Diffusion Model

```python
from app.diffusion_model import get_diffusion_model
import torch

device = torch.device('cpu')
diffusion = get_diffusion_model(device)

# Generate samples
samples = diffusion.sample(4, 3, 32, 32)
print(f"Generated {samples.shape[0]} images")
```

### Test Energy Model

```python
from app.energy_model import get_energy_model
import torch

device = torch.device('cpu')
energy_model = get_energy_model(device)

# Generate samples
samples = energy_model.sample(4, 3, 32, 32)
print(f"Generated {samples.shape[0]} images")
```

### Test GAN Model

```python
from app.mnist_gan_model import get_mnist_gan_generator

gan = get_mnist_gan_generator("models/generator_gan.pth")
image_base64 = gan.generate_digit(digit=5, seed=42)
print("Generated MNIST digit")
```

## Training Models (Optional)

Models work without training (will generate random patterns). To train:

```bash
# Train Diffusion Model (~10 hours on GPU, ~50 hours on CPU)
python -m app.train_diffusion

# Train Energy Model (~8 hours on GPU, ~40 hours on CPU)
python -m app.train_energy

# Train GAN (~2 hours on GPU, ~10 hours on CPU)
python -m app.train_mnist_gan
```

Training is optional - API works with untrained models.

## Troubleshooting

### Issue: Import errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Issue: spaCy model not found

```bash
# Install from local wheel
pip install en_core_web_lg-3.7.1-py3-none-any.whl

# Or download
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1-py3-none-any.whl
```

### Issue: Port 8000 in use

```bash
# Use different port
uvicorn app.main:app --port 8080

# Or find and kill process
lsof -i :8000
kill -9 <PID>
```

### Issue: Out of memory

```bash
# Reduce batch size in training scripts
# Or use CPU instead of GPU
device = torch.device('cpu')
```

## Verification Checklist

Run this checklist to ensure everything works:

- [ ] Clone repository successfully
- [ ] Install dependencies without errors
- [ ] `python verify_all_models.py` passes all tests (5/5)
- [ ] API starts without errors
- [ ] Can access http://localhost:8000/docs
- [ ] Health endpoint returns: `{"status": "healthy"}`
- [ ] Can generate images via API
- [ ] Jupyter notebook opens and runs

## Expected Output

When running `python verify_all_models.py`, you should see:

```
✅ PASSED: Package Imports
✅ PASSED: Diffusion Model
✅ PASSED: Energy-Based Model
✅ PASSED: GAN Model
✅ PASSED: FastAPI Application

Results: 5/5 tests passed

✅ ALL VERIFICATIONS PASSED!
```

## API Endpoints to Test

Visit http://localhost:8000/docs and try:

1. **GET /health** - Should return `{"status": "healthy"}`
2. **POST /generate-diffusion** - Generate images with DDPM
3. **POST /generate-energy** - Generate images with EBM
4. **POST /generate-digit** - Generate MNIST digits
5. **GET /diffusion-model-info** - Model information
6. **GET /energy-model-info** - Model information

## Docker-Specific Instructions

### First Time Setup

```bash
# Make scripts executable
chmod +x docker-build.sh validate_docker_setup.sh

# Validate
./validate_docker_setup.sh

# Build
./docker-build.sh
```

### Subsequent Runs

```bash
# Start
docker-compose up -d

# Stop
docker-compose down

# Rebuild
docker-compose up --build -d
```

## Files You Need

Minimum files for basic operation:
- `requirements.txt` - Python dependencies
- `app/*.py` - All application files
- `en_core_web_lg-3.7.1-py3-none-any.whl` - spaCy model

Optional files:
- `models/*.pth` - Pre-trained models (will work without these)
- `Dockerfile`, `docker-compose.yml` - For Docker deployment

## Support

If you encounter any issues:

1. **Check verification**: `python verify_all_models.py`
2. **View logs**: Check terminal output for errors
3. **Check dependencies**: `pip list | grep torch`
4. **Test imports**: `python -c "import torch; import fastapi; print('OK')"`

## Summary

To run this project:

```bash
# 1. Clone
git clone https://github.com/yuanmingchen922/genai.git
cd genai

# 2. Install
pip install -r requirements.txt

# 3. Verify
python verify_all_models.py

# 4. Run
uvicorn app.main:app --reload

# 5. Test
python app/test_assignment4_api.py
```

That's it! The project is designed to work out of the box.

## What if Something Doesn't Work?

1. Check Python version: `python --version` (should be 3.10+)
2. Update pip: `pip install --upgrade pip`
3. Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`
4. Run verification: `python verify_all_models.py`
5. Check error messages carefully

---

**Last Updated**: November 23, 2025
**Version**: 4.0.0
**Status**: Production Ready ✅

