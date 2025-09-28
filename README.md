# GenAI - Generative AI Course Assignments

This repository contains all assignments, projects, and practical implementations for the **Generative AI Course**. It serves as a comprehensive collection of hands-on learning materials covering fundamental to advanced concepts in generative artificial intelligence.

## 📚 Course Overview

This repository is structured to support progressive learning in generative AI, covering:

- **Fundamentals of Generative AI**: Basic concepts and mathematical foundations
- **Statistical Language Models**: N-gram models and probability distributions
- **Word Embeddings**: Semantic representation and similarity calculations
- **Neural Language Models**: Modern deep learning approaches
- **API Development**: Creating production-ready AI services

## 🗂️ Repository Structure

```
genai/
├── app/                          # Production FastAPI Application
│   ├── main.py                   # API routes and endpoints
│   └── bigram_model.py           # Core ML models and business logic
├── module2/                      # Module 2: Basics of Generative AI
│   ├── Module_2_Practical_1_Probability.py
│   ├── Module_2_Practical_2_Word_Sampling.py
│   ├── Module_2_Practical_3_Word_Embeddings.py
│   └── *.ipynb                   # Jupyter notebooks for interactive learning
├── module3/                      # Module 3: Advanced Applications
│   ├── Agentic_workflow.ipynb    # Agent-based AI workflows
│   └── openai_API.py             # OpenAI API integration examples
├── pyproject.toml                # Project dependencies and configuration
└── README.md                     # This documentation
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- UV package manager (recommended) or pip

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yuanmingchen922/genai.git
   cd genai
   ```

2. **Install dependencies**:
   ```bash
   # Using UV (recommended)
   uv install
   
   # Or using pip
   pip install -r requirements.txt
   ```

3. **Download required models**:
   ```bash
   uv run python -m spacy download en_core_web_lg
   ```

## 📝 Assignment Modules

### Module 2: Basics of Generative AI
**Focus**: Statistical foundations and traditional NLP approaches

- **Practical 1**: Probability distributions and language modeling
- **Practical 2**: Word sampling and text generation techniques  
- **Practical 3**: Word embeddings and semantic similarity

**Key Learning Outcomes**:
- Understanding probability in language models
- Implementing bigram and n-gram models
- Working with word vectors and semantic spaces

### Module 3: Advanced Applications
**Focus**: Modern AI applications and API development

- **Agentic Workflows**: Building intelligent agent systems
- **API Integration**: Working with OpenAI and other AI services
- **Production Deployment**: FastAPI application development

**Key Learning Outcomes**:
- Building production-ready AI applications
- Integrating multiple AI services
- Creating RESTful APIs for AI models

## 🛠️ Running the Applications

### Interactive Jupyter Notebooks
```bash
# Run Module 2 Word Embeddings (Marimo)
uv run python module2/Module_2_Practical_3_Word_Embeddings.py

# Access interactive interface at http://localhost:2718
```

### FastAPI Production Server
```bash
# Start the API server
uv run uvicorn app.main:app --reload

# Access API at http://127.0.0.1:8000
# View documentation at http://127.0.0.1:8000/docs
```

## 🔧 API Endpoints

The production FastAPI application provides the following endpoints:

| Endpoint | Method | Description |
|----------|---------|-------------|
| `/` | GET | API information and available endpoints |
| `/generate` | POST | Generate text using Bigram model |
| `/embedding` | POST | Get word embedding vectors |
| `/similarity` | POST | Calculate word-to-word similarity |
| `/sentence-similarity` | POST | Calculate query-to-sentences similarity |

### Example API Usage

```bash
# Get word embedding
curl -X POST "http://127.0.0.1:8000/embedding" \
  -H "Content-Type: application/json" \
  -d '{"word": "artificial", "return_size": 10}'

# Calculate similarity
curl -X POST "http://127.0.0.1:8000/similarity" \
  -H "Content-Type: application/json" \
  -d '{"word1": "AI", "word2": "intelligence"}'
```

## 🧪 Technologies & Libraries

- **FastAPI**: Modern, fast web framework for building APIs
- **spaCy**: Industrial-strength natural language processing
- **scikit-learn**: Machine learning library for similarity calculations
- **Marimo**: Interactive Python notebooks
- **Pydantic**: Data validation using Python type annotations
- **uvicorn**: ASGI server for FastAPI applications

## 📋 Assignment Submission Guidelines

1. **Code Quality**: Follow PEP 8 standards and include docstrings
2. **Documentation**: Update README for any new features or modules
3. **Testing**: Include test cases for critical functionality
4. **Reproducibility**: Ensure all dependencies are properly documented

## 🎯 Learning Objectives

By the end of this course, students will be able to:

- ✅ Understand mathematical foundations of generative AI
- ✅ Implement statistical language models from scratch
- ✅ Work with modern word embedding techniques
- ✅ Build production-ready AI APIs
- ✅ Integrate multiple AI services and models
- ✅ Deploy and maintain AI applications

## 📧 Course Information

- **Course**: Generative AI Fundamentals
- **Repository**: https://github.com/yuanmingchen922/genai
- **Language**: Python 3.8+
- **License**: Educational Use

## 🤝 Contributing

This repository is primarily for coursework, but suggestions and improvements are welcome:

1. Fork the repository
2. Create a feature branch
3. Make your improvements
4. Submit a pull request

---

**Note**: This repository is maintained for educational purposes as part of a Generative AI course curriculum.