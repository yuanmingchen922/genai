"""
Helper Library Usage Example
Demonstrates how to use all helper modules together for training and evaluating neural networks.
"""

import torch
import torch.nn as nn
import torch.optim as optim

# Import helper modules
from helper.data_loader import load_fashion_mnist, get_dataset_info
from helper.GAN_model import SimpleNN, SimpleCNN
from helper.trainer import train_model
from helper.evaluator import evaluate_model


def main():
    """Main function demonstrating the complete workflow."""
    
    print("="*70)
    print("HELPER LIBRARY USAGE EXAMPLE - Fashion-MNIST Classification")
    print("="*70)
    
    # 1. Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[1] Using device: {device}")
    
    # 2. Load dataset
    print("\n[2] Loading Fashion-MNIST dataset...")
    train_loader = load_fashion_mnist(batch_size=64, train=True)
    test_loader = load_fashion_mnist(batch_size=64, train=False)
    
    # Get dataset information
    dataset_info = get_dataset_info('fashion_mnist')
    print(f"    Dataset: {dataset_info['name']}")
    print(f"    Classes: {dataset_info['num_classes']}")
    print(f"    Input shape: {dataset_info['input_shape']}")
    print(f"    Class names: {', '.join(dataset_info['classes'][:3])}...")
    
    # 3. Create model
    print("\n[3] Creating model...")
    
    # Option A: Simple Fully Connected Neural Network
    model_type = "SimpleNN"  # Change to "SimpleCNN" to use CNN
    
    if model_type == "SimpleNN":
        # Flatten 28x28 image to 784 input features
        model = SimpleNN(
            layer_sizes=[784, 256, 128, 10],  # Input -> Hidden -> Hidden -> Output
            activation='relu',
            dropout_rate=0.2
        )
        print(f"    Model: SimpleNN with layers [784, 256, 128, 10]")
    else:
        # CNN for image classification
        model = SimpleCNN(
            num_classes=10,
            input_channels=1  # Grayscale images
        )
        print(f"    Model: SimpleCNN for image classification")
    
    model = model.to(device)
    print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. Setup training
    print("\n[4] Setting up training...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(f"    Loss function: CrossEntropyLoss")
    print(f"    Optimizer: Adam (lr=0.001)")
    
    # 5. Train model
    print("\n[5] Training model...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=5,
        verbose=True
    )
    
    # Print training summary
    print("\n[6] Training Summary:")
    print(f"    Final training loss: {history['train_loss'][-1]:.4f}")
    print(f"    Final training accuracy: {history['train_accuracy'][-1]:.2f}%")
    
    # 6. Evaluate model
    print("\n[7] Evaluating model on test set...")
    results = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        verbose=True
    )
    
    # 7. Save model
    print("\n[8] Saving model...")
    model_path = 'fashion_mnist_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"    Model saved to: {model_path}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Test Accuracy: {results['accuracy']:.2f}%")
    print(f"Test Precision: {results['precision']:.4f}")
    print(f"Test Recall: {results['recall']:.4f}")
    print(f"Test F1 Score: {results['f1_score']:.4f}")
    print("="*70)


def example_inference():
    """Example showing how to load and use a trained model for inference."""
    
    print("\n" + "="*70)
    print("INFERENCE EXAMPLE")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load test data
    test_loader = load_fashion_mnist(batch_size=1, train=False)
    
    # Create model and load saved weights
    model = SimpleNN([784, 256, 128, 10])
    model.load_state_dict(torch.load('fashion_mnist_model.pth'))
    model.to(device)
    model.eval()
    
    # Get one sample
    data_iter = iter(test_loader)
    image, label = next(data_iter)
    image, label = image.to(device), label.to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    # Get class names
    dataset_info = get_dataset_info('fashion_mnist')
    classes = dataset_info['classes']
    
    print(f"True label: {classes[label.item()]}")
    print(f"Predicted: {classes[predicted.item()]}")
    print("="*70)


if __name__ == "__main__":
    # Run main training example
    main()
    
    # Uncomment to run inference example
    # example_inference()
