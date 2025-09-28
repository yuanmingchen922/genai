from pydantic import BaseModel
import spacy
from typing import List, Optional

class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

class WordEmbeddingRequest(BaseModel):
    word: str
    return_size: Optional[int] = 10 

class SimilarityRequest(BaseModel):
    word1: str
    word2: str

class SentenceSimilarityRequest(BaseModel):
    query: str
    sentences: List[str]

class BigramModel:
    def __init__(self, corpus):
        """
        Initialize the BigramModel with the provided corpus
        """
        self.bigrams = {}
        for sentence in corpus:
            words = sentence.lower().split()
            for i in range(len(words) - 1):
                if words[i] not in self.bigrams:
                    self.bigrams[words[i]] = []
                self.bigrams[words[i]].append(words[i + 1])
    
    def generate_text(self, start_word, length):
        """
        Generate text based on bigram probabilities
        """
        import random
        
        result = [start_word.lower()]
        current_word = start_word.lower()
        
        for _ in range(length - 1):
            if current_word in self.bigrams:
                next_word = random.choice(self.bigrams[current_word])
                result.append(next_word)
                current_word = next_word
            else:
                break
        
        return ' '.join(result)

# Sample corpus for the bigram model
corpus = [
    "The lord of the rings.",
    "The tale of Kharis Graim is a novel written by Alexander Dumas",
    "It tells the story of Edmund Dantes, who is falsely imprisoned and later seeks rev",
    "This is another example sentence",
    "We are generating text based on bigram probabilities",
    "Bigram models are simple but effective"
]

bigram_model = BigramModel(corpus)

class WordEmbeddingModel:
    def __init__(self):
        """Initialize the WordEmbeddingModel with spaCy's large English model"""
        try:
            self.nlp = spacy.load("en_core_web_lg")
        except OSError:
            raise RuntimeError(
                "spaCy model 'en_core_web_lg' not found. "
                "Please install it using: python -m spacy download en_core_web_lg"
            )
    
    def calculate_embedding(self, input_word: str, return_size: int = 10):
        """Calculate word embedding for a given word"""
        word = self.nlp(input_word)
        full_vector = word.vector.tolist()
        
        if return_size and return_size > 0:
            return full_vector[:return_size]
        return full_vector
    
    def calculate_similarity(self, word1: str, word2: str) -> float:
        """Calculate similarity between two words"""
        return self.nlp(word1).similarity(self.nlp(word2))
    
    def calculate_sentence_similarity(self, query: str, sentences: List[str]):
        """Calculate similarity between a query and multiple sentences"""
        query_doc = self.nlp(query)
        similarities = []
        
        for sentence in sentences:
            sentence_doc = self.nlp(sentence)
            similarity = query_doc.similarity(sentence_doc)
            similarities.append({
                "sentence": sentence,
                "similarity": similarity
            })
        
        # Sort by similarity desc
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities

word_embedding_model = WordEmbeddingModel()