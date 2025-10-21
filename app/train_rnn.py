"""
Script to train the RNN text generation model
Run this before deploying the API to have a trained model
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.rnn_model import RNNTextGenerator

def main():
    print("Initializing RNN Text Generator...")
    generator = RNNTextGenerator()
    
    print("\nStarting training...")
    print("This will download 'The Count of Monte Cristo' and train the model.")
    print("Training may take several minutes depending on your hardware.\n")
    
    # Train the model
    generator.train_from_text(epochs=15)
    
    # Save the trained model
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "rnn_text_generator.pth")
    vocab_path = os.path.join(model_dir, "rnn_vocab.pkl")
    
    generator.save_model(model_path, vocab_path)
    
    # Test the model
    print("\n" + "="*50)
    print("Testing the trained model...")
    print("="*50 + "\n")
    
    test_seeds = [
        "the count of monte cristo",
        "once upon a time",
        "the quick brown"
    ]
    
    for seed in test_seeds:
        print(f"\nSeed: '{seed}'")
        generated = generator.generate_text(seed, length=30)
        print(f"Generated: {generated}\n")
        print("-" * 50)
    
    print("\n✅ Training completed successfully!")
    print(f"Model saved to: {model_path}")
    print(f"Vocabulary saved to: {vocab_path}")

if __name__ == "__main__":
    main()
