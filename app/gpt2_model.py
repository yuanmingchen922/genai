"""
GPT2 Fine-tuned Model for Question Answering

This module provides a fine-tuned GPT2 model for generating responses
in a specific format using the SQuAD dataset.

Response Format:
- Start: "That is a great question! "
- End: " Let me know if you have any other questions."
"""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from typing import Optional, Dict, Any
import os


class GPT2QAModel:
    """
    Fine-tuned GPT2 model for question-answering with custom response format.
    
    The model is trained to generate responses that:
    1. Start with "That is a great question!"
    2. Provide the answer
    3. End with "Let me know if you have any other questions."
    """
    
    # Response format templates
    RESPONSE_PREFIX = "That is a great question! "
    RESPONSE_SUFFIX = " Let me know if you have any other questions."
    
    def __init__(
        self, 
        model_name: str = "openai-community/gpt2",
        device: Optional[torch.device] = None,
        max_length: int = 150
    ):
        """
        Initialize the GPT2 QA model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Torch device for computation
            max_length: Maximum generation length
        """
        self.model_name = model_name
        self.max_length = max_length
        
        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
        
        # Load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        # Add padding token (GPT2 doesn't have one by default)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)
        
        # Track if model is fine-tuned
        self.is_fine_tuned = False
    
    def format_training_example(self, question: str, answer: str) -> str:
        """
        Format a QA pair for training with custom response format.
        
        Args:
            question: The input question
            answer: The answer to the question
        
        Returns:
            Formatted training string
        """
        formatted = (
            f"Question: {question}\n"
            f"Answer: {self.RESPONSE_PREFIX}{answer}{self.RESPONSE_SUFFIX}"
        )
        return formatted
    
    def generate_response(
        self, 
        question: str, 
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """
        Generate a response to a question.
        
        Args:
            question: The input question
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            do_sample: Whether to use sampling
        
        Returns:
            Generated response with custom format
        """
        self.model.eval()
        
        # Format input
        prompt = f"Question: {question}\nAnswer: {self.RESPONSE_PREFIX}"
        
        # Tokenize
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer part
        if "Answer: " in generated_text:
            response = generated_text.split("Answer: ")[1]
        else:
            response = generated_text
        
        # Ensure proper format
        if not response.startswith(self.RESPONSE_PREFIX):
            response = self.RESPONSE_PREFIX + response
        
        if not response.endswith(self.RESPONSE_SUFFIX):
            # Find a good ending point
            if "." in response and not response.endswith("."):
                last_period = response.rfind(".")
                response = response[:last_period + 1]
            response = response.rstrip() + self.RESPONSE_SUFFIX
        
        return response
    
    def save_model(self, save_path: str):
        """
        Save the fine-tuned model.
        
        Args:
            save_path: Directory path to save the model
        """
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save metadata
        metadata = {
            "model_name": self.model_name,
            "is_fine_tuned": self.is_fine_tuned,
            "response_prefix": self.RESPONSE_PREFIX,
            "response_suffix": self.RESPONSE_SUFFIX
        }
        torch.save(metadata, os.path.join(save_path, "metadata.pt"))
    
    def load_model(self, load_path: str):
        """
        Load a fine-tuned model.
        
        Args:
            load_path: Directory path to load the model from
        """
        self.model = GPT2LMHeadModel.from_pretrained(load_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(load_path)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.to(self.device)
        
        # Load metadata if exists
        metadata_path = os.path.join(load_path, "metadata.pt")
        if os.path.exists(metadata_path):
            metadata = torch.load(metadata_path, map_location='cpu')
            self.is_fine_tuned = metadata.get("is_fine_tuned", True)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary containing model information
        """
        num_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_name": self.model_name,
            "total_parameters": num_params,
            "trainable_parameters": trainable_params,
            "is_fine_tuned": self.is_fine_tuned,
            "device": str(self.device),
            "max_length": self.max_length,
            "vocab_size": len(self.tokenizer),
            "response_format": {
                "prefix": self.RESPONSE_PREFIX,
                "suffix": self.RESPONSE_SUFFIX
            }
        }


# Global model instance cache
_gpt2_model_cache = {}


def get_gpt2_model(
    model_path: Optional[str] = None,
    device: Optional[torch.device] = None
) -> GPT2QAModel:
    """
    Get or create a GPT2 QA model instance.
    
    Args:
        model_path: Path to fine-tuned model (optional)
        device: Torch device
    
    Returns:
        GPT2QAModel instance
    """
    cache_key = model_path or "default"
    
    if cache_key not in _gpt2_model_cache:
        model = GPT2QAModel(device=device)
        
        if model_path and os.path.exists(model_path):
            model.load_model(model_path)
        
        _gpt2_model_cache[cache_key] = model
    
    return _gpt2_model_cache[cache_key]

