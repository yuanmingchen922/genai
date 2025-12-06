from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel, Field
from .bigram_model import (
    TextGenerationRequest, 
    WordEmbeddingRequest,
    SimilarityRequest,
    SentenceSimilarityRequest,
    bigram_model, 
    word_embedding_model
)
from .rnn_model import rnn_generator
from .cnn_classifier import (
    ImageClassificationRequest,
    get_classifier
)
from .mnist_gan_model import get_mnist_gan_generator
from .diffusion_model import get_diffusion_model
from .energy_model import get_energy_model
from .gpt2_model import get_gpt2_model
import base64
from typing import Optional
import torch

app = FastAPI(
    title="GenAI API",
    description="API for text generation (Bigram, RNN, Fine-tuned GPT2), word embeddings, image classification (CNN), and image generation (GAN, Diffusion, Energy)",
    version="5.0.0"
)

@app.get("/")
def read_root():
    return {
        "message": "GenAI API - Text Processing, Image Classification, and Image Generation",
        "endpoints": {
            "text": [
                "/generate", 
                "/generate_with_rnn",
                "/generate-gpt2",
                "/embedding", 
                "/similarity", 
                "/sentence-similarity",
                "/gpt2-model-info"
            ],
            "image_classification": [
                "/classify-image", 
                "/classify-image-file"
            ],
            "image_generation": [
                "/generate-digit",
                "/generate-digits-batch",
                "/gan-model-info",
                "/generate-diffusion",
                "/generate-energy",
                "/diffusion-model-info",
                "/energy-model-info"
            ]
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint for Docker container monitoring"""
    return {
        "status": "healthy",
        "service": "genai-api",
        "version": "5.0.0"
    }

@app.post("/generate")
async def generate_text(request: TextGenerationRequest):
    """Generate text using Bigram model"""
    try:
        generated_text = bigram_model.generate_text(request.start_word, request.length)
        return {"generated_text": generated_text, "start_word": request.start_word, "length": request.length}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text generation failed: {str(e)}")


@app.post("/generate_with_rnn")
async def generate_with_rnn(request: TextGenerationRequest):
    """Generate text using RNN/LSTM model"""
    try:
        generated_text = rnn_generator.generate_text(
            seed_text=request.start_word, 
            length=request.length,
            temperature=1.0
        )
        return {
            "generated_text": generated_text, 
            "start_word": request.start_word, 
            "length": request.length,
            "model": "LSTM"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RNN text generation failed: {str(e)}")

@app.post("/embedding")
async def get_word_embedding(request: WordEmbeddingRequest):
    """Get word embedding vector for a given word"""
    try:
        embedding = word_embedding_model.calculate_embedding(request.word, request.return_size)
        return {
            "word": request.word,
            "embedding": embedding,
            "dimensions_returned": len(embedding),
            "total_dimensions": 300
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Word embedding calculation failed: {str(e)}")

@app.post("/similarity")
async def calculate_word_similarity(request: SimilarityRequest):
    """Calculate similarity between two words"""
    try:
        similarity = word_embedding_model.calculate_similarity(request.word1, request.word2)
        return {
            "word1": request.word1,
            "word2": request.word2,
            "similarity": similarity
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Similarity calculation failed: {str(e)}")

@app.post("/sentence-similarity")
async def calculate_sentence_similarity(request: SentenceSimilarityRequest):
    """Calculate similarity between a query and multiple sentences"""
    try:
        similarities = word_embedding_model.calculate_sentence_similarity(request.query, request.sentences)
        return {
            "query": request.query,
            "results": similarities,
            "total_sentences": len(request.sentences)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentence similarity calculation failed: {str(e)}")


@app.post("/classify-image")
async def classify_image(request: ImageClassificationRequest):
    """
    Classify an image using the trained CNN model
    
    The image should be base64 encoded and sent in the request body.
    Returns the top predicted classes with confidence scores.
    """
    try:
        classifier = get_classifier(model_path="models/cnn_classifier.pth")
        result = classifier.predict(request.image_data, top_k=request.top_k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image classification failed: {str(e)}")


@app.post("/classify-image-file")
async def classify_image_file(file: UploadFile = File(...), top_k: int = 1):
    """
    Classify an image uploaded as a file
    
    Upload an image file (JPEG, PNG, etc.) and get classification results.
    Returns the top predicted classes with confidence scores.
    """
    try:
        # Read file contents
        contents = await file.read()
        
        # Convert to base64
        image_base64 = base64.b64encode(contents).decode('utf-8')
        
        # Get classifier and predict
        classifier = get_classifier(model_path="models/cnn_classifier.pth")
        result = classifier.predict(image_base64, top_k=top_k)
        
        return {
            "filename": file.filename,
            **result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image classification failed: {str(e)}")


# ============================================================================
# GAN Image Generation Endpoints
# ============================================================================

class DigitGenerationRequest(BaseModel):
    """Request model for digit generation"""
    digit: Optional[int] = Field(None, ge=0, le=9, description="Specific digit to generate (0-9). If None, generates random digit.")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")


class BatchGenerationRequest(BaseModel):
    """Request model for batch digit generation"""
    batch_size: int = Field(16, ge=1, le=64, description="Number of digits to generate")
    grid: bool = Field(True, description="Return images as a grid or list")


@app.post("/generate-digit")
async def generate_digit(request: DigitGenerationRequest):
    """
    Generate a single MNIST-style hand-written digit using GAN.
    
    Args:
        digit: Optional specific digit (0-9). If None, generates random digit.
        seed: Optional random seed for reproducibility.
    
    Returns:
        Base64 encoded PNG image of the generated digit.
    """
    try:
        gan_generator = get_mnist_gan_generator(model_path="models/generator_gan.pth")
        image_base64 = gan_generator.generate_digit(digit=request.digit, seed=request.seed)
        
        return {
            "success": True,
            "image": image_base64,
            "format": "base64_png",
            "size": "28x28",
            "requested_digit": request.digit,
            "seed": request.seed,
            "note": "Without conditional GAN, specific digit cannot be guaranteed"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Digit generation failed: {str(e)}")


@app.post("/generate-digits-batch")
async def generate_digits_batch(request: BatchGenerationRequest):
    """
    Generate multiple MNIST-style hand-written digits using GAN.
    
    Args:
        batch_size: Number of digits to generate (1-64)
        grid: If True, returns images arranged in a grid. If False, returns list of individual images.
    
    Returns:
        Base64 encoded PNG image(s) of the generated digits.
    """
    try:
        gan_generator = get_mnist_gan_generator(model_path="models/generator_gan.pth")
        
        if request.grid:
            grid_image = gan_generator.generate_batch(batch_size=request.batch_size, grid=True)
            return {
                "success": True,
                "image": grid_image,
                "format": "base64_png",
                "batch_size": request.batch_size,
                "layout": "grid"
            }
        else:
            images = gan_generator.generate_batch(batch_size=request.batch_size, grid=False)
            return {
                "success": True,
                "images": images,
                "format": "base64_png",
                "batch_size": request.batch_size,
                "count": len(images)
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch generation failed: {str(e)}")


@app.get("/gan-model-info")
async def get_gan_model_info():
    """
    Get information about the loaded MNIST GAN model.
    
    Returns:
        Model architecture and parameter information.
    """
    try:
        gan_generator = get_mnist_gan_generator(model_path="models/generator_gan.pth")
        model_info = gan_generator.get_model_info()
        
        return {
            "success": True,
            **model_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


# ============================================================================
# Diffusion Model Endpoints
# ============================================================================

class DiffusionGenerationRequest(BaseModel):
    """Request model for diffusion generation"""
    num_samples: int = Field(8, ge=1, le=64, description="Number of images to generate")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")


@app.post("/generate-diffusion")
async def generate_diffusion(request: DiffusionGenerationRequest):
    """
    Generate images using Diffusion Model (DDPM).
    
    Generates CIFAR-10 style images (32x32 RGB) using denoising diffusion.
    
    Args:
        num_samples: Number of images to generate (1-64)
        seed: Optional random seed for reproducibility
    
    Returns:
        Base64 encoded PNG images of generated samples.
    """
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 
                            'mps' if torch.backends.mps.is_available() else 'cpu')
        
        # Load diffusion model
        diffusion = get_diffusion_model(device)
        
        # Load trained weights if available
        try:
            checkpoint = torch.load("models/diffusion_cifar10_final.pth", map_location=device)
            diffusion.model.load_state_dict(checkpoint['model_state_dict'])
        except:
            # If no trained model, use untrained (will generate random patterns)
            pass
        
        # Set seed if provided
        if request.seed is not None:
            torch.manual_seed(request.seed)
        
        # Generate samples
        samples = diffusion.sample(request.num_samples, 3, 32, 32)
        
        # Convert to base64
        # Denormalize from [-1, 1] to [0, 1]
        samples = (samples + 1) / 2
        samples = torch.clamp(samples, 0, 1)
        
        # Create grid
        from torchvision.utils import make_grid
        from PIL import Image
        import io
        
        grid = make_grid(samples, nrow=4, padding=2)
        grid_np = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
        
        # Convert to base64
        img = Image.fromarray(grid_np)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            "success": True,
            "image": img_base64,
            "format": "base64_png",
            "num_samples": request.num_samples,
            "image_size": "32x32",
            "model": "DDPM (Denoising Diffusion Probabilistic Model)",
            "seed": request.seed
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diffusion generation failed: {str(e)}")


@app.get("/diffusion-model-info")
async def get_diffusion_model_info():
    """
    Get information about the Diffusion Model.
    
    Returns:
        Model architecture and parameter information.
    """
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 
                            'mps' if torch.backends.mps.is_available() else 'cpu')
        
        diffusion = get_diffusion_model(device)
        
        num_params = sum(p.numel() for p in diffusion.model.parameters())
        
        return {
            "success": True,
            "model_type": "Denoising Diffusion Probabilistic Model (DDPM)",
            "architecture": "UNet with Time Embeddings",
            "parameters": num_params,
            "timesteps": diffusion.timesteps,
            "beta_schedule": "linear",
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "image_size": "32x32",
            "channels": 3,
            "dataset": "CIFAR-10",
            "device": str(device)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get diffusion model info: {str(e)}")


# ============================================================================
# Energy-Based Model Endpoints
# ============================================================================

class EnergyGenerationRequest(BaseModel):
    """Request model for energy-based generation"""
    num_samples: int = Field(8, ge=1, le=64, description="Number of images to generate")
    langevin_steps: Optional[int] = Field(60, ge=10, le=200, description="Number of Langevin sampling steps")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")


@app.post("/generate-energy")
async def generate_energy(request: EnergyGenerationRequest):
    """
    Generate images using Energy-Based Model with Langevin Sampling.
    
    Generates CIFAR-10 style images (32x32 RGB) using energy minimization.
    
    Args:
        num_samples: Number of images to generate (1-64)
        langevin_steps: Number of Langevin dynamics steps (10-200)
        seed: Optional random seed for reproducibility
    
    Returns:
        Base64 encoded PNG images of generated samples.
    """
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 
                            'mps' if torch.backends.mps.is_available() else 'cpu')
        
        # Load energy model
        energy_model = get_energy_model(device, use_improved=True)
        
        # Update Langevin steps if provided
        if request.langevin_steps:
            energy_model.langevin_steps = request.langevin_steps
        
        # Load trained weights if available
        try:
            checkpoint = torch.load("models/energy_cifar10_final.pth", map_location=device)
            energy_model.model.load_state_dict(checkpoint['model_state_dict'])
        except:
            # If no trained model, use untrained
            pass
        
        # Set seed if provided
        if request.seed is not None:
            torch.manual_seed(request.seed)
        
        # Generate samples via Langevin sampling
        samples = energy_model.sample(request.num_samples, 3, 32, 32)
        
        # Convert to base64
        # Denormalize from [-1, 1] to [0, 1]
        samples = (samples + 1) / 2
        samples = torch.clamp(samples, 0, 1)
        
        # Create grid
        from torchvision.utils import make_grid
        from PIL import Image
        import io
        
        grid = make_grid(samples, nrow=4, padding=2)
        grid_np = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
        
        # Convert to base64
        img = Image.fromarray(grid_np)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            "success": True,
            "image": img_base64,
            "format": "base64_png",
            "num_samples": request.num_samples,
            "image_size": "32x32",
            "model": "Energy-Based Model with Langevin Sampling",
            "langevin_steps": energy_model.langevin_steps,
            "seed": request.seed
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Energy-based generation failed: {str(e)}")


@app.get("/energy-model-info")
async def get_energy_model_info():
    """
    Get information about the Energy-Based Model.
    
    Returns:
        Model architecture and parameter information.
    """
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 
                            'mps' if torch.backends.mps.is_available() else 'cpu')
        
        energy_model = get_energy_model(device, use_improved=True)
        
        num_params = sum(p.numel() for p in energy_model.model.parameters())
        
        return {
            "success": True,
            "model_type": "Energy-Based Model (EBM)",
            "architecture": "Convolutional Energy Network with Spectral Normalization",
            "parameters": num_params,
            "sampling_method": "Langevin Dynamics",
            "langevin_steps": energy_model.langevin_steps,
            "langevin_lr": energy_model.langevin_lr,
            "langevin_noise": energy_model.langevin_noise,
            "image_size": "32x32",
            "channels": 3,
            "dataset": "CIFAR-10",
            "training_method": "Contrastive Divergence",
            "device": str(device)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get energy model info: {str(e)}")


# ============================================================================
# Fine-tuned GPT2 Text Generation Endpoints
# ============================================================================

class GPT2GenerationRequest(BaseModel):
    """Request model for GPT2 text generation"""
    question: str = Field(..., description="The question to answer")
    max_new_tokens: int = Field(100, ge=10, le=500, description="Maximum number of tokens to generate")
    temperature: float = Field(0.7, ge=0.1, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.1, le=1.0, description="Nucleus sampling probability")


@app.post("/generate-gpt2")
async def generate_with_gpt2(request: GPT2GenerationRequest):
    """
    Generate a response using fine-tuned GPT2 model.
    
    The model is fine-tuned on SQuAD dataset to generate responses in a specific format:
    - Starts with: "That is a great question!"
    - Ends with: "Let me know if you have any other questions."
    
    Args:
        question: The input question
        max_new_tokens: Maximum tokens to generate (10-500)
        temperature: Sampling temperature (0.1-2.0)
        top_p: Nucleus sampling probability (0.1-1.0)
    
    Returns:
        Generated response with the formatted answer.
    """
    try:
        # Load model (uses cache)
        model_path = "models/gpt2_finetuned"
        gpt2_model = get_gpt2_model(model_path=model_path)
        
        # Generate response
        response = gpt2_model.generate_response(
            question=request.question,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        return {
            "success": True,
            "question": request.question,
            "response": response,
            "model": "GPT2 (Fine-tuned on SQuAD)",
            "parameters": {
                "max_new_tokens": request.max_new_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GPT2 generation failed: {str(e)}")


@app.get("/gpt2-model-info")
async def get_gpt2_model_info():
    """
    Get information about the fine-tuned GPT2 model.
    
    Returns:
        Model architecture and fine-tuning information.
    """
    try:
        model_path = "models/gpt2_finetuned"
        gpt2_model = get_gpt2_model(model_path=model_path)
        
        model_info = gpt2_model.get_model_info()
        
        return {
            "success": True,
            **model_info,
            "fine_tuning_dataset": "SQuAD (Stanford Question Answering Dataset)",
            "fine_tuning_source": "https://huggingface.co/datasets/rajpurkar/squad"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get GPT2 model info: {str(e)}")
