"""
Evaluator Module
Provides evaluation functions for neural network models.
"""

import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np


def evaluate_model(model, test_loader, criterion, device, verbose=True):
    """
    Evaluate a neural network model on test data.
    
    Args:
        model (nn.Module): PyTorch model to evaluate
        test_loader (DataLoader): DataLoader for test data
        criterion: Loss function (e.g., nn.CrossEntropyLoss())
        device (str or torch.device): Device to evaluate on ('cuda' or 'cpu')
        verbose (bool): If True, print evaluation results. Default: True
    
    Returns:
        dict: Dictionary containing evaluation metrics:
              - 'loss': Average loss on test set
              - 'accuracy': Overall accuracy (%)
              - 'precision': Precision score (macro average)
              - 'recall': Recall score (macro average)
              - 'f1_score': F1 score (macro average)
              - 'confusion_matrix': Confusion matrix
    
    Example:
        >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        >>> results = evaluate_model(model, test_loader, criterion, device)
        >>> print(f"Test Accuracy: {results['accuracy']:.2f}%")
    """
    model.to(device)
    model.eval()
    
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            # Store predictions and labels
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    avg_loss = running_loss / len(test_loader)
    accuracy = 100. * np.sum(all_predictions == all_labels) / len(all_labels)
    precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    
    results = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix
    }
    
    if verbose:
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Loss: {avg_loss:.4f}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("="*50 + "\n")
    
    return results


def calculate_accuracy(model, data_loader, device):
    """
    Calculate accuracy of model on a dataset.
    
    Args:
        model (nn.Module): PyTorch model
        data_loader (DataLoader): DataLoader for data
        device (str or torch.device): Device to evaluate on
    
    Returns:
        float: Accuracy percentage
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy


def get_predictions(model, data_loader, device):
    """
    Get predictions and true labels for a dataset.
    
    Args:
        model (nn.Module): PyTorch model
        data_loader (DataLoader): DataLoader for data
        device (str or torch.device): Device to evaluate on
    
    Returns:
        tuple: (predictions, true_labels) as numpy arrays
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels)


__all__ = ['evaluate_model', 'calculate_accuracy', 'get_predictions']
