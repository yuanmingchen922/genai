# GenAI - Advanced Generative AI Project

A comprehensive FastAPI-based application for text generation and image generation using various deep learning models.

## Features

- **Text Generation**: Bigram, RNN/LSTM, and Fine-tuned GPT2 models
- **Image Classification**: CNN classifier for CIFAR-10
- **Image Generation**: GAN, Diffusion, and Energy-Based Models
- **Question Answering**: GPT2 fine-tuned on SQuAD dataset

## API Version: 5.0.0

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run API
uvicorn app.main:app --reload

# Access docs
http://localhost:8000/docs
```

## API Endpoints

### Text Generation
- `POST /generate` - Bigram model
- `POST /generate_with_rnn` - RNN/LSTM model
- `POST /generate-gpt2` - Fine-tuned GPT2

### Image Generation
- `POST /generate-digit` - GAN digit generation
- `POST /generate-diffusion` - Diffusion model
- `POST /generate-energy` - Energy-based model

## Assignment History

- **Task 2**: MNIST GAN Implementation
- **Task 4**: Diffusion and Energy-Based Models on CIFAR-10
- **Task 5**: Fine-tuned GPT2 for QA with custom response format

