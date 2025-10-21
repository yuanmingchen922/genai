import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import re
import requests
from collections import Counter
from typing import Optional
import os
import pickle

class LSTMModel(nn.Module):
    """LSTM-based text generation model"""
    def __init__(self, vocab_size=10000, embedding_dim=100, hidden_dim=128):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        x, hidden = self.lstm(x, hidden)
        x = self.fc(x)
        return x, hidden


class TextDataset(Dataset):
    """Dataset for text sequences"""
    def __init__(self, data, seq_len=30):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx:idx+self.seq_len]),
            torch.tensor(self.data[idx+1:idx+self.seq_len+1])
        )


class RNNTextGenerator:
    """RNN-based text generator with training and inference capabilities"""
    
    def __init__(self, model_path: Optional[str] = None, vocab_path: Optional[str] = None):
        """
        Initialize the RNN text generator
        
        Args:
            model_path: Path to saved model weights
            vocab_path: Path to saved vocabulary
        """
        self.vocab = None
        self.inv_vocab = None
        self.model = None
        self.seq_len = 30
        
        if model_path and vocab_path and os.path.exists(model_path) and os.path.exists(vocab_path):
            self.load_model(model_path, vocab_path)
        else:
            # Initialize with default vocabulary and model
            self._initialize_default()
    
    def _initialize_default(self):
        """Initialize with a basic vocabulary and untrained model"""
        # Create minimal vocabulary
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        self.inv_vocab = {0: "<PAD>", 1: "<UNK>"}
        
        # Initialize model
        self.model = LSTMModel(vocab_size=10000, embedding_dim=100, hidden_dim=128)
        self.model.eval()
    
    def train_from_text(self, text_url: str = None, text_content: str = None, epochs: int = 15):
        """
        Train the model from text data
        
        Args:
            text_url: URL to download text from
            text_content: Direct text content (alternative to URL)
            epochs: Number of training epochs
        """
        # Load text
        if text_url:
            text = requests.get(text_url).text
        elif text_content:
            text = text_content
        else:
            # Default: Load Count of Monte Cristo
            url = "https://www.gutenberg.org/cache/epub/1184/pg1184.txt"
            text = requests.get(url).text
            
            # Extract main body
            start_idx = text.find("Chapter 1.")
            end_idx = text.rfind("Chapter 5.")
            text = text[start_idx:end_idx] if start_idx != -1 and end_idx != -1 else text
        
        # Preprocess
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        text = text.lower()
        
        # Tokenization
        tokens = text.split()
        
        # Build vocabulary
        counter = Counter(tokens)
        self.vocab = {word: idx+2 for idx, (word, _) in enumerate(counter.most_common(9998))}
        self.vocab["<PAD>"] = 0
        self.vocab["<UNK>"] = 1
        self.inv_vocab = {idx: word for word, idx in self.vocab.items()}
        
        # Encode tokens
        encoded = [self.vocab.get(word, self.vocab["<UNK>"]) for word in tokens]
        
        # Create dataset and dataloader
        train_dataset = TextDataset(encoded, self.seq_len)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        # Initialize model
        vocab_size = len(self.vocab)
        self.model = LSTMModel(vocab_size=vocab_size, embedding_dim=100, hidden_dim=128)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters())
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for inputs, targets in train_loader:
                outputs, _ = self.model(inputs)
                loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
        
        self.model.eval()
        print("Training completed!")
    
    def generate_text(self, seed_text: str, length: int = 50, temperature: float = 1.0) -> str:
        """
        Generate text based on seed text
        
        Args:
            seed_text: Starting text for generation
            length: Number of words to generate
            temperature: Sampling temperature (higher = more random)
            
        Returns:
            Generated text string
        """
        if self.model is None or self.vocab is None:
            return f"{seed_text} [Model not trained]"
        
        self.model.eval()
        words = seed_text.lower().split()
        input_ids = [self.vocab.get(w, self.vocab["<UNK>"]) for w in words]
        input_tensor = torch.tensor(input_ids).unsqueeze(0)
        hidden = None
        
        with torch.no_grad():
            for _ in range(length):
                output, hidden = self.model(input_tensor, hidden)
                logits = output[0, -1] / temperature
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).item()
                words.append(self.inv_vocab.get(next_id, "<UNK>"))
                
                # Extend input sequence with new token
                input_ids.append(next_id)
                input_tensor = torch.tensor(input_ids).unsqueeze(0)
        
        return " ".join(words)
    
    def save_model(self, model_path: str, vocab_path: str):
        """
        Save model weights and vocabulary
        
        Args:
            model_path: Path to save model weights
            vocab_path: Path to save vocabulary
        """
        if self.model is not None:
            torch.save(self.model.state_dict(), model_path)
        
        if self.vocab is not None:
            with open(vocab_path, 'wb') as f:
                pickle.dump({
                    'vocab': self.vocab,
                    'inv_vocab': self.inv_vocab
                }, f)
        
        print(f"Model saved to {model_path}")
        print(f"Vocabulary saved to {vocab_path}")
    
    def load_model(self, model_path: str, vocab_path: str):
        """
        Load model weights and vocabulary
        
        Args:
            model_path: Path to model weights
            vocab_path: Path to vocabulary
        """
        # Load vocabulary
        with open(vocab_path, 'rb') as f:
            vocab_data = pickle.load(f)
            self.vocab = vocab_data['vocab']
            self.inv_vocab = vocab_data['inv_vocab']
        
        # Initialize and load model
        vocab_size = len(self.vocab)
        self.model = LSTMModel(vocab_size=vocab_size, embedding_dim=100, hidden_dim=128)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"Vocabulary loaded from {vocab_path}")


# Initialize global RNN text generator
# Automatically load pre-trained model if available
model_path = "models/rnn_text_generator.pth"
vocab_path = "models/rnn_vocab.pkl"

if os.path.exists(model_path) and os.path.exists(vocab_path):
    print(f"Loading pre-trained RNN model from {model_path}")
    rnn_generator = RNNTextGenerator(model_path=model_path, vocab_path=vocab_path)
    print("✅ RNN model loaded successfully!")
else:
    print("⚠️  No pre-trained RNN model found. Initialize with untrained model.")
    print(f"To train the model, run: python -m app.train_rnn")
    rnn_generator = RNNTextGenerator()

# You can manually train by calling: rnn_generator.train_from_text()
