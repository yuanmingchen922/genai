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
import base64
from typing import Optional

app = FastAPI(
    title="GenAI API",
    description="API for text generation (Bigram & RNN), word embeddings, image classification (CNN), and image generation (GAN)",
    version="3.0.0"
)

@app.get("/")
def read_root():
    return {
        "message": "GenAI API - Text Processing, Image Classification, and Image Generation",
        "endpoints": {
            "text": [
                "/generate", 
                "/generate_with_rnn",
                "/embedding", 
                "/similarity", 
                "/sentence-similarity"
            ],
            "image_classification": [
                "/classify-image", 
                "/classify-image-file"
            ],
            "image_generation": [
                "/generate-digit",
                "/generate-digits-batch",
                "/gan-model-info"
            ]
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint for Docker container monitoring"""
    return {
        "status": "healthy",
        "service": "genai-api",
        "version": "3.0.0"
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
