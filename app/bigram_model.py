from pydantic import BaseModel

class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

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