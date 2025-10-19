"""
Training script for CNN Image Classifier
Trains the Enhanced CNN on CIFAR-10 dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.cnn_classifier import AssignmentCNN


def train_cnn_model(
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.0005,
    save_path: str = "models/cnn_classifier.pth",
    device: str = None
):
    """
    Train the CNN model on CIFAR-10 dataset
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        save_path: Path to save the trained model
        device: Device to train on ('cpu', 'cuda', or 'mps')
    """
    
    # Set device
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device)
    
    print(f"Training on device: {device}")
    
    # Prepare the data with augmentation
    # Resize CIFAR-10 images from 32x32 to 64x64 to match assignment spec
    transform_train = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize to 64x64 per assignment
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(64, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize to 64x64 per assignment
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load CIFAR-10 dataset
    print("Loading CIFAR-10 dataset...")
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform_train
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform_test
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Initialize model - Use AssignmentCNN as per assignment specification
    model = AssignmentCNN(num_classes=10).to(device)
    print(f"\nModel architecture:\n{model}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=2,
        verbose=True
    )
    
    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_loader_progress = tqdm(
            train_loader, 
            desc=f'Epoch {epoch+1}/{epochs} [Train]'
        )
        
        for batch_idx, (inputs, labels) in enumerate(train_loader_progress):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            train_loader_progress.set_postfix({
                'loss': f'{running_loss/(batch_idx+1):.3f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100. * correct / total
        
        # Validation phase
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            test_loader_progress = tqdm(
                test_loader, 
                desc=f'Epoch {epoch+1}/{epochs} [Test]'
            )
            
            for inputs, labels in test_loader_progress:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                test_loader_progress.set_postfix({
                    'loss': f'{test_loss/(len(test_loader_progress)):.3f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        test_loss = test_loss / len(test_loader)
        test_accuracy = 100. * correct / total
        
        # Update learning rate scheduler
        scheduler.step(test_loss)
        
        # Print epoch summary
        print(f'\nEpoch {epoch+1}/{epochs} Summary:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'  Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%')
        
        # Save best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f'  ✓ New best model saved with accuracy: {best_accuracy:.2f}%')
        
        print('-' * 70)
    
    print(f"\nTraining completed!")
    print(f"Best test accuracy: {best_accuracy:.2f}%")
    print(f"Model saved to: {save_path}")
    
    return model, best_accuracy


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train CNN Image Classifier')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--save-path', type=str, default='models/cnn_classifier.pth', 
                       help='Path to save model')
    parser.add_argument('--device', type=str, default=None, 
                       help='Device to train on (cpu/cuda/mps)')
    
    args = parser.parse_args()
    
    train_cnn_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_path=args.save_path,
        device=args.device
    )
