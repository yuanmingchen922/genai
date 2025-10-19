"""
CNN Image Classifier Module
Implements a Convolutional Neural Network for CIFAR-10 image classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from PIL import Image
import io
import base64


class ImageClassificationRequest(BaseModel):
    """Request model for image classification"""
    image_data: str  # Base64 encoded image
    top_k: Optional[int] = 1  # Return top K predictions


class ImageClassificationResponse(BaseModel):
    """Response model for image classification"""
    predictions: List[dict]
    confidence_scores: List[float]


class AssignmentCNN(nn.Module):
    """
    Convolutional Neural Network for CIFAR-10 classification
    Implements the architecture specified in the assignment:
    
    Input: RGB image of size 64×64×3
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
    
    def __init__(self, num_classes=10):
        super(AssignmentCNN, self).__init__()
        
        # Convolutional Layer 1: 3 -> 16 channels
        # Input: (batch, 3, 64, 64) -> Output: (batch, 16, 64, 64)
        self.conv1 = nn.Conv2d(
            in_channels=3, 
            out_channels=16, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )
        
        # MaxPooling Layer 1
        # Input: (batch, 16, 64, 64) -> Output: (batch, 16, 32, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional Layer 2: 16 -> 32 channels
        # Input: (batch, 16, 32, 32) -> Output: (batch, 32, 32, 32)
        self.conv2 = nn.Conv2d(
            in_channels=16, 
            out_channels=32, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )
        
        # MaxPooling Layer 2
        # Input: (batch, 32, 32, 32) -> Output: (batch, 32, 16, 16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # After 2 pooling layers: 64x64 -> 32x32 -> 16x16
        # Flattened size: 32 * 16 * 16 = 8192
        
        # Fully connected layer 1: 8192 -> 100
        self.fc1 = nn.Linear(32 * 16 * 16, 100)
        
        # Fully connected layer 2: 100 -> 10
        self.fc2 = nn.Linear(100, num_classes)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, 3, 64, 64)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Conv Block 1: Conv -> ReLU -> MaxPool
        x = self.conv1(x)           # (batch, 16, 64, 64)
        x = F.relu(x)               # (batch, 16, 64, 64)
        x = self.pool1(x)           # (batch, 16, 32, 32)
        
        # Conv Block 2: Conv -> ReLU -> MaxPool
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


# Keep the enhanced version for backwards compatibility
class EnhancedCNN(nn.Module):
    """
    Enhanced Convolutional Neural Network for CIFAR-10 classification
    
    Architecture:
    - 4 Convolutional layers with BatchNorm and ReLU activation
    - MaxPooling after each conv layer
    - 2 Fully connected layers with Dropout
    - Output layer with 10 classes
    
    This architecture is based on the Module 4 practical exercises.
    """
    
    def __init__(self, num_classes=10):
        super(EnhancedCNN, self).__init__()
        
        # Convolutional Layer 1: 3 -> 16 channels
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Convolutional Layer 2: 16 -> 32 channels
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Convolutional Layer 3: 32 -> 64 channels
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Convolutional Layer 4: 64 -> 128 channels
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        # After 4 pooling layers: 32x32 -> 16x16 -> 8x8 -> 4x4 -> 2x2
        self.fc1 = nn.Linear(128 * 2 * 2, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Conv Block 1: Conv -> BatchNorm -> ReLU -> Pool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Conv Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Conv Block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Conv Block 4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(-1, 128 * 2 * 2)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class CNNImageClassifier:
    """
    Image classifier using the Enhanced CNN model
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
        
        # Initialize model - Use AssignmentCNN for the assignment specification
        self.model = AssignmentCNN(num_classes=10)
        
        # Load weights if provided
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model weights from {model_path}")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Define image preprocessing - Resize to 64x64 as per assignment spec
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),  # Resize to 64x64 per assignment
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


# Global classifier instance (will be initialized when needed)
cnn_classifier = None


def get_classifier(model_path: Optional[str] = None):
    """Get or create the global CNN classifier instance"""
    global cnn_classifier
    if cnn_classifier is None:
        cnn_classifier = CNNImageClassifier(model_path=model_path)
    return cnn_classifier
