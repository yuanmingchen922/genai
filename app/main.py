from fastapi import FastAPI, HTTPException, File, UploadFile
from .bigram_model import (
    TextGenerationRequest, 
    WordEmbeddingRequest,
    SimilarityRequest,
    SentenceSimilarityRequest,
    bigram_model, 
    word_embedding_model
)
from .cnn_classifier import (
    ImageClassificationRequest,
    get_classifier
)
import base64

app = FastAPI(
    title="GenAI API",
    description="API for text generation, word embeddings, and image classification using CNNs",
    version="2.0.0"
)

@app.get("/")
def read_root():
    return {
        "message": "GenAI API - Text Processing and Image Classification",
        "endpoints": {
            "text": ["/generate", "/embedding", "/similarity", "/sentence-similarity"],
            "image": ["/classify-image", "/classify-image-file"]
        }
    }

@app.post("/generate")
async def generate_text(request: TextGenerationRequest):
    """Generate text using Bigram model"""
    try:
        generated_text = bigram_model.generate_text(request.start_word, request.length)
        return {"generated_text": generated_text, "start_word": request.start_word, "length": request.length}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text generation failed: {str(e)}")

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
