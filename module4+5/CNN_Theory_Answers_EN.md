# CNN Theory Questions - Detailed Answers
**Student**: [my2878]  
**Date**: 2025-10-19

## CNN Output Size Calculation Formula

For convolutional and pooling layers, the output size is calculated using:

$$
\text{Output Size} = \left\lfloor \frac{\text{Input Size} - \text{Kernel Size} + 2 \times \text{Padding}}{\text{Stride}} \right\rfloor + 1
$$

Where:
- Input Size: Size of input feature map
- Kernel Size: Size of convolution/pooling kernel
- Padding: Padding size
- Stride: Stride value

---

## Question 1
**Question**: Given an input image of size 32×32×3 and a convolutional layer with 8 filters of size 5×5, stride 1, and no padding, what is the output size?

### Answer:

**Formula Application**:
- Input Size = 32
- Kernel Size = 5
- Stride = 1
- Padding = 0

$$
\text{Output Size} = \left\lfloor \frac{32 - 5 + 2 \times 0}{1} \right\rfloor + 1 = \left\lfloor \frac{27}{1} \right\rfloor + 1 = 27 + 1 = 28
$$

**Answer**: Output size is **28×28×8**

**Explanation**:
- Spatial dimensions change from 32×32 to 28×28 (due to convolution without padding)
- Channels change from 3 (RGB) to 8 (number of filters)
- Each filter produces one output channel

---

## Question 2
**Question**: How does the output size change if padding is changed to "same"?

### Answer:

**Meaning of "same" padding**:
"same" padding means the output spatial dimensions equal the input dimensions (when stride=1).

To maintain output size of 32×32, calculate required padding:

$$
\text{Output Size} = \left\lfloor \frac{\text{Input Size} - \text{Kernel Size} + 2 \times \text{Padding}}{\text{Stride}} \right\rfloor + 1 = \text{Input Size}
$$

For kernel size = 5, stride = 1:

$$
32 = \left\lfloor \frac{32 - 5 + 2 \times \text{Padding}}{1} \right\rfloor + 1
$$

$$
31 = 32 - 5 + 2 \times \text{Padding}
$$

$$
2 \times \text{Padding} = 4 \quad \Rightarrow \quad \text{Padding} = 2
$$

**Answer**: Output size is **32×32×8**

**Explanation**:
- "same" padding adds 2 pixels around the image (2 pixels on each side)
- This preserves spatial dimensions
- Channel count remains 8 (number of filters)

---

## Question 3
**Question**: If you apply a 3×3 filter with stride 2 and no padding to a 64×64 input, what is the output spatial size?

### Answer:

**Formula Application**:
- Input Size = 64
- Kernel Size = 3
- Stride = 2
- Padding = 0

$$
\text{Output Size} = \left\lfloor \frac{64 - 3 + 2 \times 0}{2} \right\rfloor + 1 = \left\lfloor \frac{61}{2} \right\rfloor + 1 = 30 + 1 = 31
$$

**Answer**: Output spatial size is **31×31**

**Explanation**:
- Stride of 2 means the filter moves 2 pixels at a time
- This approximately halves the spatial dimensions
- Output size = (64-3)/2 + 1 = 31

---

## Question 4
**Question**: You apply a max-pooling layer of size 2×2 with stride 2 on a 16×16 feature map. What is the output size?

### Answer:

**Formula Application** (max pooling typically has no padding):
- Input Size = 16
- Kernel Size = 2
- Stride = 2
- Padding = 0

$$
\text{Output Size} = \left\lfloor \frac{16 - 2 + 2 \times 0}{2} \right\rfloor + 1 = \left\lfloor \frac{14}{2} \right\rfloor + 1 = 7 + 1 = 8
$$

**Answer**: Output size is **8×8**

**Explanation**:
- Max pooling layer halves the spatial dimensions of the feature map
- 2×2 pooling window with stride 2 reduces 16×16 to 8×8
- Channel count remains unchanged (pooling doesn't affect channels)
- Pooling reduces feature map size by selecting maximum values from local regions

---

## Question 5
**Question**: An image of shape 128×128 is passed through two successive convolutional layers. Each uses a 3×3 kernel, stride 1, and 'same' padding. What is the output shape?

### Answer:

**First Convolutional Layer**:
- Input: 128×128
- Kernel: 3×3, stride=1, padding="same"
- Output: 128×128 ("same" padding preserves dimensions)

**Second Convolutional Layer**:
- Input: 128×128
- Kernel: 3×3, stride=1, padding="same"
- Output: 128×128 ("same" padding preserves dimensions)

**Answer**: Output shape is **128×128** (spatial dimensions)

**Explanation**:
- "same" padding ensures output spatial dimensions equal input when stride=1
- For 3×3 kernel, "same" padding requires padding=1
- Two successive "same" convolutions both preserve spatial dimensions
- Channel count depends on number of filters in each layer (not specified), but spatial dimensions remain 128×128

**Mathematical Verification** (padding=1):
$$
\text{Output Size} = \left\lfloor \frac{128 - 3 + 2 \times 1}{1} \right\rfloor + 1 = \left\lfloor \frac{127}{1} \right\rfloor + 1 = 128
$$

---

## Question 6
**Question**: In the examples in class, before starting the training loop we ran: `model.train()`. What happens if you remove that line?

### Answer:

**Purpose of `model.train()`**:

`model.train()` sets the model to **training mode**, which is crucial for certain layers:

1. **Dropout Layers**:
   - Training mode: Randomly drops neurons (according to dropout probability)
   - Eval mode: Keeps all neurons, no dropping

2. **Batch Normalization Layers**:
   - Training mode: Uses current batch mean/variance for normalization, updates running statistics
   - Eval mode: Uses accumulated running mean/variance from training

3. **Other layers** (like DropConnect, Layer Normalization) may also have different behaviors

**What Happens if You Remove `model.train()`?**

**Scenario Analysis**:

- **After model initialization**: Model is in training mode by default, removing this line may have no immediate effect
- **After calling `model.eval()`**: If you previously called `model.eval()` (for validation/testing), then forget to call `model.train()` before training, this causes:
  
  1. **Dropout won't activate**: All neurons are kept, leading to:
     - Loss of regularization effect
     - Possible overfitting
     - Inconsistent behavior between training and testing
  
  2. **Batch Normalization behavior is wrong**: Uses fixed statistics instead of current batch statistics, causing:
     - Incorrect normalization
     - Wrong gradient calculations
     - Model fails to learn properly
     - Significantly degraded training performance

**Answer**: If you remove `model.train()`, especially after `model.eval()`, it causes:
- **Dropout layers don't work**, losing regularization
- **Batch Normalization uses wrong statistics**, causing training failure
- **Model may fail to learn or converge properly**
- **Inconsistent behavior between training and inference**

**Best Practice**:
```python
# Training phase
model.train()  # Set to training mode
for epoch in range(num_epochs):
    for batch in train_loader:
        # Training code
        ...

# Validation/Test phase
model.eval()  # Set to evaluation mode
with torch.no_grad():  # Disable gradient computation
    for batch in val_loader:
        # Validation code
        ...
```

---

## Summary

| Question | Answer | Key Concept |
|----------|--------|-------------|
| Q1 | 28×28×8 | Basic convolution output calculation |
| Q2 | 32×32×8 | "same" padding preserves dimensions |
| Q3 | 31×31 | Stride=2 approximately halves dimensions |
| Q4 | 8×8 | Pooling reduces feature map size |
| Q5 | 128×128 | Successive "same" convolutions preserve size |
| Q6 | Training fails | Dropout/BN behavior mode switching |

---

## Appendix: Python Verification Code

```python
import torch
import torch.nn as nn

# Question 1: Verify output size
input_q1 = torch.randn(1, 3, 32, 32)
conv_q1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=1, padding=0)
output_q1 = conv_q1(input_q1)
print(f"Q1 Output shape: {output_q1.shape}")  # torch.Size([1, 8, 28, 28])

# Question 2: "same" padding
conv_q2 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=1, padding=2)
output_q2 = conv_q2(input_q1)
print(f"Q2 Output shape: {output_q2.shape}")  # torch.Size([1, 8, 32, 32])

# Question 3: Stride=2
input_q3 = torch.randn(1, 1, 64, 64)
conv_q3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=0)
output_q3 = conv_q3(input_q3)
print(f"Q3 Output shape: {output_q3.shape}")  # torch.Size([1, 1, 31, 31])

# Question 4: Max Pooling
input_q4 = torch.randn(1, 1, 16, 16)
pool_q4 = nn.MaxPool2d(kernel_size=2, stride=2)
output_q4 = pool_q4(input_q4)
print(f"Q4 Output shape: {output_q4.shape}")  # torch.Size([1, 1, 8, 8])

# Question 5: Two successive "same" convolutions
input_q5 = torch.randn(1, 3, 128, 128)
conv1_q5 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)  # "same"
conv2_q5 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # "same"
output_q5 = conv2_q5(conv1_q5(input_q5))
print(f"Q5 Output shape: {output_q5.shape}")  # torch.Size([1, 32, 128, 128])

# Question 6: model.train() vs model.eval()
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x

model = SimpleModel()
x = torch.randn(4, 3, 32, 32)

# Training mode
model.train()
output_train = model(x)
print(f"Q6 Train mode - Dropout active, BN uses batch stats")

# Evaluation mode
model.eval()
output_eval = model(x)
print(f"Q6 Eval mode - Dropout inactive, BN uses running stats")
print(f"Outputs are different: {not torch.allclose(output_train, output_eval)}")
```

**Expected Output**:
```
Q1 Output shape: torch.Size([1, 8, 28, 28])
Q2 Output shape: torch.Size([1, 8, 32, 32])
Q3 Output shape: torch.Size([1, 1, 31, 31])
Q4 Output shape: torch.Size([1, 1, 8, 8])
Q5 Output shape: torch.Size([1, 32, 128, 128])
Q6 Train mode - Dropout active, BN uses batch stats
Q6 Eval mode - Dropout inactive, BN uses running stats
Outputs are different: True
```

---

**All answers verified with PyTorch calculations!**

For interactive verification, run:
```bash
python module4+5/verify_cnn_theory.py
```
