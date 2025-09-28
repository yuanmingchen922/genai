from fastapi import FastAPI, HTTPException
from .bigram_model import (
    TextGenerationRequest, 
    WordEmbeddingRequest,
    SimilarityRequest,
    SentenceSimilarityRequest,
    bigram_model, 
    word_embedding_model
)

app = FastAPI(
    title="GenAI Text Processing API",
    description="API for text generation using Bigram models and word embeddings using spaCy",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {"message": "GenAI Text Processing API", "endpoints": ["/generate", "/embedding", "/similarity", "/sentence-similarity"]}

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