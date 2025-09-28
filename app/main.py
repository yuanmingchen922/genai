from fastapi import FastAPI
from .bigram_model import TextGenerationRequest, bigram_model

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate")
async def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}