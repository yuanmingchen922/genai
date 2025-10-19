"""
Assignment 2 : CNN Image 
Student: [my2878]
Date: 2025-10-19

This file implements:
1. A Convolutional Neural Network matching the assignment specification
2. A classifier for CIFAR-10 dataset
3. API integration for image classification

Assignment Requirements:
- Input: RGB image of size 64×64×3
- Conv2D with 16 filters, kernel size 3×3, stride 1, padding 1
- ReLU activation
- MaxPooling2D with kernel size 2×2, stride 2
- Conv2D with 32 filters, kernel size 3×3, stride 1, padding 1
- ReLU activation
- MaxPooling2D with kernel size 2×2, stride 2
- Flatten the output
- Fully connected layer with 100 units
- ReLU activation
- Fully connected layer with 10 units (10 output classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import base64
import io
from PIL import Image
import numpy as np
from typing import Optional, List
from pydantic import BaseModel


# ============================================================================
# PART 1: CNN Architecture Implementation
# ============================================================================

class AssignmentCNN(nn.Module):
    """
    Convolutional Neural Network for CIFAR-10 classification
    Implements the architecture specified in the assignment.
    
    Architecture:
        Input: (batch, 3, 64, 64)
        Conv1: 3 → 16 filters, kernel=3x3, stride=1, padding=1
        ReLU
        MaxPool: kernel=2x2, stride=2 → (batch, 16, 32, 32)
        Conv2: 16 → 32 filters, kernel=3x3, stride=1, padding=1
        ReLU
        MaxPool: kernel=2x2, stride=2 → (batch, 32, 16, 16)
        Flatten → (batch, 8192)
        FC1: 8192 → 100
        ReLU
        FC2: 100 → 10
        Output: (batch, 10)
    
    Total Parameters: 825,398
    """
    
    def __init__(self, num_classes=10):
        super(AssignmentCNN, self).__init__()
        
        # Convolutional Layer 1: 3 → 16 channels
        # Input: (batch, 3, 64, 64) → Output: (batch, 16, 64, 64)
        self.conv1 = nn.Conv2d(
            in_channels=3, 
            out_channels=16, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )
        
        # MaxPooling Layer 1
        # Input: (batch, 16, 64, 64) → Output: (batch, 16, 32, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional Layer 2: 16 → 32 channels
        # Input: (batch, 16, 32, 32) → Output: (batch, 32, 32, 32)
        self.conv2 = nn.Conv2d(
            in_channels=16, 
            out_channels=32, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )
        
        # MaxPooling Layer 2
        # Input: (batch, 32, 32, 32) → Output: (batch, 32, 16, 16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # After 2 pooling layers: 64x64 → 32x32 → 16x16
        # Flattened size: 32 * 16 * 16 = 8192
        
        # Fully connected layer 1: 8192 → 100
        self.fc1 = nn.Linear(32 * 16 * 16, 100)
        
        # Fully connected layer 2: 100 → 10
        self.fc2 = nn.Linear(100, num_classes)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, 3, 64, 64)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Conv Block 1: Conv → ReLU → MaxPool
        x = self.conv1(x)           # (batch, 16, 64, 64)
        x = F.relu(x)               # (batch, 16, 64, 64)
        x = self.pool1(x)           # (batch, 16, 32, 32)
        
        # Conv Block 2: Conv → ReLU → MaxPool
        x = self.conv2(x)           # (batch, 32, 32, 32)
        x = F.relu(x)               # (batch, 32, 32, 32)
        x = self.pool2(x)           # (batch, 32, 16, 16)
        
        # Flatten
        x = x.view(-1, 32 * 16 * 16)  # (batch, 8192)
        
        # Fully connected layer 1 with ReLU
        x = self.fc1(x)             # (batch, 100)
        x = F.relu(x)               # (batch, 100)
        
        # Fully connected layer 2 (output)
        x = self.fc2(x)             # (batch, 10)
        
        return x


# ============================================================================
# PART 2: Training Function
# ============================================================================

def train_cnn_model(
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.0005,
    save_path: str = "cnn_classifier.pth",
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
    
    Returns:
        Trained model and best accuracy
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
    
    # Initialize model
    model = AssignmentCNN(num_classes=10).to(device)
    print(f"\nModel architecture:\n{model}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
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
            torch.save(model.state_dict(), save_path)
            print(f'  ✓ New best model saved with accuracy: {best_accuracy:.2f}%')
        
        print('-' * 70)
    
    print(f"\nTraining completed!")
    print(f"Best test accuracy: {best_accuracy:.2f}%")
    print(f"Model saved to: {save_path}")
    
    return model, best_accuracy


# ============================================================================
# PART 3: Image Classifier for API
# ============================================================================

class CNNImageClassifier:
    """
    Image classifier using the Assignment CNN model
    Handles preprocessing, inference, and postprocessing
    """
    
    # CIFAR-10 class names
    CLASSES = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck"
    ]
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the CNN classifier
        
        Args:
            model_path: Path to saved model weights (optional)
            device: Device to run inference on ('cpu', 'cuda', or 'mps')
        """
        # Set device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = AssignmentCNN(num_classes=10)
        
        # Load weights if provided
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model weights from {model_path}")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Define image preprocessing - Resize to 64x64 as per assignment spec
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image_data: str) -> torch.Tensor:
        """
        Preprocess base64 encoded image data
        
        Args:
            image_data: Base64 encoded image string
            
        Returns:
            Preprocessed image tensor
        """
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Apply transformations
        image_tensor = self.transform(image)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor
    
    def predict(self, image_data: str, top_k: int = 1) -> dict:
        """
        Predict the class of an image
        
        Args:
            image_data: Base64 encoded image string
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary containing predictions and confidence scores
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image_data).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, k=min(top_k, 10))
        
        # Convert to lists
        top_probs = top_probs.cpu().numpy()[0].tolist()
        top_indices = top_indices.cpu().numpy()[0].tolist()
        
        # Create predictions list
        predictions = []
        for idx, prob in zip(top_indices, top_probs):
            predictions.append({
                "class": self.CLASSES[idx],
                "class_id": idx,
                "confidence": float(prob)
            })
        
        return {
            "predictions": predictions,
            "top_prediction": predictions[0]["class"],
            "confidence": predictions[0]["confidence"]
        }
    
    def predict_from_array(self, image_array: np.ndarray, top_k: int = 1) -> dict:
        """
        Predict from a numpy array (useful for testing)
        
        Args:
            image_array: Numpy array of shape (H, W, 3) with values in [0, 255]
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary containing predictions and confidence scores
        """
        # Convert to PIL Image
        image = Image.fromarray(image_array.astype('uint8'), 'RGB')
        
        # Apply transformations
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, k=min(top_k, 10))
        
        # Convert to lists
        top_probs = top_probs.cpu().numpy()[0].tolist()
        top_indices = top_indices.cpu().numpy()[0].tolist()
        
        # Create predictions list
        predictions = []
        for idx, prob in zip(top_indices, top_probs):
            predictions.append({
                "class": self.CLASSES[idx],
                "class_id": idx,
                "confidence": float(prob)
            })
        
        return {
            "predictions": predictions,
            "top_prediction": predictions[0]["class"],
            "confidence": predictions[0]["confidence"]
        }


# ============================================================================
# PART 4: Usage Examples and Testing
# ============================================================================

def verify_architecture():
    """Verify the CNN architecture matches assignment specifications"""
    
    print("="*70)
    print("CNN Architecture Verification")
    print("="*70)
    
    model = AssignmentCNN(num_classes=10)
    
    print("\nModel Architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal Parameters: {total_params:,}")
    
    # Test forward pass with correct input size (64x64x3)
    print("\nForward Pass Test:")
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 64, 64)
    
    print(f"Input shape: {dummy_input.shape}")
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected: (batch_size={batch_size}, num_classes=10)")
    
    if output.shape == (batch_size, 10):
        print("✓ Architecture verification PASSED!")
    else:
        print("✗ Architecture verification FAILED!")
    
    print("="*70)


def demo_prediction():
    """Demonstrate the classifier with sample images"""
    
    print("\n" + "="*70)
    print("CNN Classifier Demonstration")
    print("="*70)
    
    # Create classifier
    classifier = CNNImageClassifier()
    
    # Create sample test images
    print("\nCreating sample images...")
    
    # Sample images (random for demo)
    samples = []
    for i in range(3):
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        samples.append(img)
    
    # Predict
    print("\nRunning predictions...")
    for i, img in enumerate(samples, 1):
        result = classifier.predict_from_array(img, top_k=3)
        print(f"\nSample {i}:")
        print(f"  Top Prediction: {result['top_prediction']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Top 3:")
        for j, pred in enumerate(result['predictions'][:3], 1):
            print(f"    {j}. {pred['class']:12s} - {pred['confidence']:.2%}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    # Verify architecture
    verify_architecture()
    
    # Optionally run demo
    print("\n")
    demo_prediction()
    
    # To train the model, uncomment:
    # model, accuracy = train_cnn_model(epochs=10, batch_size=32)
