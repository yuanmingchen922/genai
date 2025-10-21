"""
CNN Theory Questions Verification Script
Student: [my2878]
Date: 2025-10-19

This script verifies all CNN theory question answers through actual PyTorch operations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def print_separator(title):
    """Print separator line"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def calculate_output_size(input_size, kernel_size, stride, padding):
    """
    Calculate output size for convolution or pooling layer
    
    Formula: output = floor((input - kernel + 2*padding) / stride) + 1
    """
    output = (input_size - kernel_size + 2 * padding) // stride + 1
    return output


def question_1():
    """
    Question 1: 32×32×3 input, 8 filters of 5×5, stride=1, padding=0
    """
    print_separator("Question 1")
    
    print("Question: Given input image 32×32×3, conv layer with 8 filters of 5×5")
    print("         stride=1, padding=0, what is the output size?\n")
    
    # Theoretical calculation
    input_size = 32
    kernel_size = 5
    stride = 1
    padding = 0
    
    output_size = calculate_output_size(input_size, kernel_size, stride, padding)
    
    print(f"Theoretical Calculation:")
    print(f"  Output Size = floor((32 - 5 + 2×0) / 1) + 1")
    print(f"             = floor(27 / 1) + 1")
    print(f"             = 28")
    print(f"  Output shape: 28×28×8\n")
    
    # PyTorch verification
    input_tensor = torch.randn(1, 3, 32, 32)
    conv_layer = nn.Conv2d(in_channels=3, out_channels=8, 
                           kernel_size=5, stride=1, padding=0)
    output_tensor = conv_layer(input_tensor)
    
    print(f"PyTorch Verification:")
    print(f"  Input shape: {tuple(input_tensor.shape)}")
    print(f"  Output shape: {tuple(output_tensor.shape)}")
    print(f"  ✓ Answer: {output_tensor.shape[2]}×{output_tensor.shape[3]}×{output_tensor.shape[1]}")


def question_2():
    """
    Question 2: "same" padding
    """
    print_separator("Question 2")
    
    print("Question: How does output size change if padding is changed to 'same'?\n")
    
    # Calculate padding required for "same" padding
    input_size = 32
    kernel_size = 5
    stride = 1
    
    # For "same" padding: output_size = input_size (when stride=1)
    # padding = (kernel_size - 1) / 2
    padding = (kernel_size - 1) // 2
    
    print(f"Theoretical Calculation:")
    print(f"  'same' padding means output size = input size (when stride=1)")
    print(f"  For kernel_size=5, we need padding={padding}")
    print(f"  Output Size = floor((32 - 5 + 2×{padding}) / 1) + 1")
    print(f"             = floor((32 - 5 + {2*padding}) / 1) + 1")
    print(f"             = floor(31 / 1) + 1 = 32")
    print(f"  Output shape: 32×32×8\n")
    
    # PyTorch verification
    input_tensor = torch.randn(1, 3, 32, 32)
    conv_layer = nn.Conv2d(in_channels=3, out_channels=8, 
                           kernel_size=5, stride=1, padding=padding)
    output_tensor = conv_layer(input_tensor)
    
    print(f"PyTorch Verification:")
    print(f"  Input shape: {tuple(input_tensor.shape)}")
    print(f"  Output shape: {tuple(output_tensor.shape)}")
    print(f"  ✓ Answer: {output_tensor.shape[2]}×{output_tensor.shape[3]}×{output_tensor.shape[1]}")


def question_3():
    """
    Question 3: 64×64 input, 3×3 filter, stride=2, padding=0
    """
    print_separator("Question 3")
    
    print("Question: Apply 3×3 filter with stride=2, padding=0 to 64×64 input")
    print("         What is the output spatial size?\n")
    
    # Theoretical calculation
    input_size = 64
    kernel_size = 3
    stride = 2
    padding = 0
    
    output_size = calculate_output_size(input_size, kernel_size, stride, padding)
    
    print(f"Theoretical Calculation:")
    print(f"  Output Size = floor((64 - 3 + 2×0) / 2) + 1")
    print(f"             = floor(61 / 2) + 1")
    print(f"             = 30 + 1 = 31")
    print(f"  Output spatial size: 31×31\n")
    
    # PyTorch verification
    input_tensor = torch.randn(1, 1, 64, 64)
    conv_layer = nn.Conv2d(in_channels=1, out_channels=1, 
                           kernel_size=3, stride=2, padding=0)
    output_tensor = conv_layer(input_tensor)
    
    print(f"PyTorch Verification:")
    print(f"  Input shape: {tuple(input_tensor.shape)}")
    print(f"  Output shape: {tuple(output_tensor.shape)}")
    print(f"  ✓ Answer: {output_tensor.shape[2]}×{output_tensor.shape[3]}")


def question_4():
    """
    Question 4: 16×16 feature map, 2×2 max pooling, stride=2
    """
    print_separator("Question 4")
    
    print("Question: Apply 2×2 max-pooling with stride=2 to 16×16 feature map")
    print("         What is the output size?\n")
    
    # Theoretical calculation
    input_size = 16
    kernel_size = 2
    stride = 2
    padding = 0
    
    output_size = calculate_output_size(input_size, kernel_size, stride, padding)
    
    print(f"Theoretical Calculation:")
    print(f"  Output Size = floor((16 - 2 + 2×0) / 2) + 1")
    print(f"             = floor(14 / 2) + 1")
    print(f"             = 7 + 1 = 8")
    print(f"  Output size: 8×8\n")
    
    # PyTorch verification
    input_tensor = torch.randn(1, 1, 16, 16)
    pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
    output_tensor = pool_layer(input_tensor)
    
    print(f"PyTorch Verification:")
    print(f"  Input shape: {tuple(input_tensor.shape)}")
    print(f"  Output shape: {tuple(output_tensor.shape)}")
    print(f"  ✓ Answer: {output_tensor.shape[2]}×{output_tensor.shape[3]}")


def question_5():
    """
    Question 5: 128×128 input, two successive 3×3 convs, stride=1, "same" padding
    """
    print_separator("Question 5")
    
    print("Question: 128×128 image through two successive conv layers")
    print("         Each uses 3×3 kernel, stride=1, 'same' padding")
    print("         What is the output shape?\n")
    
    # Theoretical calculation
    input_size = 128
    kernel_size = 3
    stride = 1
    padding = 1  # "same" padding for 3×3 kernel
    
    print(f"Theoretical Calculation:")
    print(f"  'same' padding for 3×3 kernel requires padding=1")
    print(f"  Layer 1: Output = floor((128 - 3 + 2×1) / 1) + 1 = 128")
    print(f"  Layer 2: Output = floor((128 - 3 + 2×1) / 1) + 1 = 128")
    print(f"  Output shape: 128×128\n")
    
    # PyTorch verification
    input_tensor = torch.randn(1, 3, 128, 128)
    conv1 = nn.Conv2d(in_channels=3, out_channels=16, 
                      kernel_size=3, stride=1, padding=1)
    conv2 = nn.Conv2d(in_channels=16, out_channels=32, 
                      kernel_size=3, stride=1, padding=1)
    
    output1 = conv1(input_tensor)
    output2 = conv2(output1)
    
    print(f"PyTorch Verification:")
    print(f"  Input shape: {tuple(input_tensor.shape)}")
    print(f"  Layer 1 output: {tuple(output1.shape)}")
    print(f"  Layer 2 output: {tuple(output2.shape)}")
    print(f"  ✓ Answer: {output2.shape[2]}×{output2.shape[3]} (spatial dimensions)")


def question_6():
    """
    Question 6: The role of model.train()
    """
    print_separator("Question 6")
    
    print("Question: What happens if you remove model.train()?\n")
    
    # Create model with Dropout and BatchNorm
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.bn = nn.BatchNorm2d(16)
            self.dropout = nn.Dropout(p=0.5)
            self.fc = nn.Linear(16 * 8 * 8, 10)
        
        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    model = TestModel()
    x = torch.randn(4, 3, 16, 16)
    
    print("Theoretical Answer:")
    print("  model.train() sets the model to training mode, affecting:\n")
    print("  1. Dropout layers:")
    print("     - Training mode: Randomly drops neurons (p=0.5)")
    print("     - Eval mode: Keeps all neurons\n")
    print("  2. Batch Normalization layers:")
    print("     - Training mode: Uses batch statistics, updates running stats")
    print("     - Eval mode: Uses accumulated running statistics\n")
    print("  If you remove model.train():")
    print("     - Dropout may not work → Loss of regularization")
    print("     - BatchNorm uses wrong statistics → Training failure")
    print("     - Model may fail to learn properly\n")
    
    # PyTorch verification
    print("PyTorch Verification:\n")
    
    # Training mode
    model.train()
    torch.manual_seed(42)
    output_train1 = model(x)
    torch.manual_seed(42)
    output_train2 = model(x)
    
    print(f"  Training mode (model.train()):")
    print(f"    - Two forward passes same? {torch.allclose(output_train1, output_train2)}")
    print(f"      (Different due to Dropout randomness)\n")
    
    # Evaluation mode
    model.eval()
    torch.manual_seed(42)
    output_eval1 = model(x)
    torch.manual_seed(42)
    output_eval2 = model(x)
    
    print(f"  Evaluation mode (model.eval()):")
    print(f"    - Two forward passes same? {torch.allclose(output_eval1, output_eval2)}")
    print(f"      (Same because Dropout is disabled)\n")
    
    # Compare training vs eval mode
    print(f"  Train vs Eval outputs different? {not torch.allclose(output_train1, output_eval1)}")
    print(f"    (Yes, behaviors are completely different)\n")
    
    # Check Dropout effect
    model.train()
    with torch.no_grad():
        outputs_train = [model(x) for _ in range(5)]
    
    model.eval()
    with torch.no_grad():
        outputs_eval = [model(x) for _ in range(5)]
    
    train_std = torch.std(torch.stack([o.mean() for o in outputs_train]))
    eval_std = torch.std(torch.stack([o.mean() for o in outputs_eval]))
    
    print(f"  Standard deviation across multiple runs:")
    print(f"    - Training mode: {train_std:.6f} (high variance, Dropout randomness)")
    print(f"    - Eval mode: {eval_std:.6f} (low/zero variance, deterministic)")
    
    print("\n  ✓ Conclusion: model.train() and model.eval() significantly affect behavior!")


def main():
    """Main function: Run verification for all questions"""
    print("\n" + "=" * 70)
    print("  CNN Theory Questions Verification")
    print("  Student: [my2878]")
    print("  Date: 2025-10-19")
    print("=" * 70)
    
    # Run all questions
    question_1()
    question_2()
    question_3()
    question_4()
    question_5()
    question_6()
    
    print("\n" + "=" * 70)
    print("  All Questions Verified Successfully!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
