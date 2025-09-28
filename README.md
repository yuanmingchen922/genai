# GenAI Text Processing API

A comprehensive FastAPI application that combines traditional N-gram models with modern word embeddings for text processing and generation.

## Features

- **Text Generation**: Bigram-based text generation using statistical language modeling
- **Word Embeddings**: Word vector calculations using spaCy's large English model
- **Similarity Calculation**: Word and sentence similarity using semantic embeddings
- **RESTful API**: Complete FastAPI implementation with automatic documentation

## API Endpoints

- `GET /` - API information and available endpoints
- `POST /generate` - Generate text using Bigram model
- `POST /embedding` - Get word embedding vectors
- `POST /similarity` - Calculate word similarity
- `POST /sentence-similarity` - Calculate sentence similarity

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/genai.git
cd genai
```

2. Install dependencies using uv:
```bash
uv install
```

3. Download the spaCy model:
```bash
uv run python -m spacy download en_core_web_lg
```

## Usage

1. Start the FastAPI server:
```bash
uv run uvicorn app.main:app --reload
```

2. Access the API at `http://127.0.0.1:8000`
3. View interactive documentation at `http://127.0.0.1:8000/docs`

## Project Structure

```
genai/
├── app/
│   ├── main.py           # FastAPI application and routes
│   └── bigram_model.py   # Business logic and models
├── module2/              # Learning materials and examples
├── module3/              # Additional examples and notebooks  
├── pyproject.toml        # Project configuration
└── README.md            # This file
```

## Technologies Used

- **FastAPI**: Modern web framework for APIs
- **spaCy**: Natural language processing library
- **Pydantic**: Data validation and serialization
- **uvicorn**: ASGI server for FastAPI
- **scikit-learn**: Machine learning utilities

## License

This project is for educational purposes.