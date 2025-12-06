"""
GPT2 Fine-tuning Script using SQuAD Dataset

This script fine-tunes GPT2 on the Stanford Question Answering Dataset (SQuAD)
to generate responses in a specific format:
- Prefix: "That is a great question! "
- Suffix: " Let me know if you have any other questions."
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer,
    AdamW,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
from tqdm import tqdm
import os
import argparse
from typing import List, Dict


# Response format constants
RESPONSE_PREFIX = "That is a great question! "
RESPONSE_SUFFIX = " Let me know if you have any other questions."


class SQuADDataset(Dataset):
    """
    Custom dataset for SQuAD with formatted responses.
    """
    
    def __init__(
        self, 
        tokenizer: GPT2Tokenizer,
        split: str = "train",
        max_length: int = 256,
        max_samples: int = None
    ):
        """
        Initialize the dataset.
        
        Args:
            tokenizer: GPT2 tokenizer
            split: Dataset split ("train" or "validation")
            max_length: Maximum sequence length
            max_samples: Maximum number of samples to use (for quick testing)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load SQuAD dataset
        print(f"Loading SQuAD {split} dataset...")
        dataset = load_dataset("rajpurkar/squad", split=split)
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        self.examples = []
        
        print("Formatting examples...")
        for item in tqdm(dataset, desc="Processing"):
            question = item["question"]
            # Get the first answer
            answer = item["answers"]["text"][0] if item["answers"]["text"] else ""
            
            if answer:  # Only include if there's an answer
                formatted = self._format_example(question, answer)
                self.examples.append(formatted)
        
        print(f"Loaded {len(self.examples)} examples")
    
    def _format_example(self, question: str, answer: str) -> str:
        """Format a QA pair with custom response format."""
        return (
            f"Question: {question}\n"
            f"Answer: {RESPONSE_PREFIX}{answer}{RESPONSE_SUFFIX}"
        )
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        text = self.examples[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        # For language modeling, labels are the same as input_ids
        labels = input_ids.clone()
        
        # Mask padding tokens in labels
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def train_gpt2(
    output_dir: str = "models/gpt2_finetuned",
    model_name: str = "openai-community/gpt2",
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    max_length: int = 256,
    max_samples: int = None,
    warmup_steps: int = 100,
    save_steps: int = 500,
    device: str = None
):
    """
    Fine-tune GPT2 on SQuAD dataset.
    
    Args:
        output_dir: Directory to save the fine-tuned model
        model_name: HuggingFace model identifier
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        max_length: Maximum sequence length
        max_samples: Maximum number of samples (for quick testing)
        warmup_steps: Number of warmup steps
        save_steps: Save checkpoint every N steps
        device: Device to use for training
    """
    # Set device
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    
    # Load tokenizer
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    # Add padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print(f"Loading model from {model_name}...")
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    
    # Create dataset
    train_dataset = SQuADDataset(
        tokenizer=tokenizer,
        split="train",
        max_length=max_length,
        max_samples=max_samples
    )
    
    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Setup scheduler
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    print("\nStarting training...")
    model.train()
    global_step = 0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        
        for batch in progress_bar:
            # Move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            epoch_loss += loss.item()
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update weights
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Save checkpoint
            if global_step % save_steps == 0:
                checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                print(f"\nSaved checkpoint to {checkpoint_dir}")
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")
    
    # Save final model
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save metadata
    metadata = {
        "model_name": model_name,
        "is_fine_tuned": True,
        "response_prefix": RESPONSE_PREFIX,
        "response_suffix": RESPONSE_SUFFIX,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_length": max_length
    }
    torch.save(metadata, os.path.join(output_dir, "metadata.pt"))
    
    print(f"\nTraining complete! Model saved to {output_dir}")
    
    return model, tokenizer


def test_generation(model_path: str = "models/gpt2_finetuned"):
    """
    Test the fine-tuned model with sample questions.
    """
    from gpt2_model import get_gpt2_model
    
    model = get_gpt2_model(model_path=model_path)
    
    test_questions = [
        "What is machine learning?",
        "Who invented the telephone?",
        "What is the capital of France?"
    ]
    
    print("\n" + "=" * 60)
    print("Testing Fine-tuned GPT2 Model")
    print("=" * 60)
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        response = model.generate_response(question)
        print(f"Response: {response}")
        print("-" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune GPT2 on SQuAD")
    parser.add_argument("--output_dir", type=str, default="models/gpt2_finetuned")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--test_only", action="store_true")
    
    args = parser.parse_args()
    
    if args.test_only:
        test_generation(args.output_dir)
    else:
        train_gpt2(
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_samples=args.max_samples
        )

