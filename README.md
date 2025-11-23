# GenAI - Advanced Generative Models API

Complete implementation of modern generative AI models with FastAPI.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python verify_all_models.py

# Run API
uvicorn app.main:app --reload
```

Visit: http://localhost:8000/docs

## Models

- **Diffusion Model (DDPM)** - CIFAR-10 image generation
- **Energy-Based Model** - Langevin sampling on CIFAR-10
- **GAN** - MNIST digit generation
- **CNN Classifier** - Image classification
- **RNN/LSTM** - Text generation

## API Endpoints

### Image Generation
- POST `/generate-diffusion` - Diffusion model
- POST `/generate-energy` - Energy-based model
- POST `/generate-digit` - GAN digit generation

### Information
- GET `/health` - Health check
- GET `/diffusion-model-info` - Model details
- GET `/energy-model-info` - Model details

Full documentation: http://localhost:8000/docs

## Assignment 4

Main submission: `Assignments/Advanced_Image_Generation_my2878.ipynb`

Includes:
- Complete theory answers (Questions 1-9)
- Full model implementations
- Training scripts for CIFAR-10

## Docker

```bash
docker-compose up --build -d
```

## Repository

https://github.com/yuanmingchen922/genai.git

Student: my2878

