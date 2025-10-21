# CNN 理论问题解答
**学生**: [my2878]  
**日期**: 2025-10-19

## CNN 输出尺寸计算公式

对于卷积层和池化层，输出尺寸的计算公式为：

$$
\text{Output Size} = \left\lfloor \frac{\text{Input Size} - \text{Kernel Size} + 2 \times \text{Padding}}{\text{Stride}} \right\rfloor + 1
$$

其中：
- Input Size：输入特征图的尺寸
- Kernel Size：卷积核或池化核的尺寸
- Padding：填充大小
- Stride：步长

---

## Question 1
**问题**: 给定输入图像大小为 32×32×3，卷积层有 8 个 5×5 的滤波器，步长为 1，无填充，输出大小是多少？

### 解答：

**公式应用**：
- Input Size = 32
- Kernel Size = 5
- Stride = 1
- Padding = 0

$$
\text{Output Size} = \left\lfloor \frac{32 - 5 + 2 \times 0}{1} \right\rfloor + 1 = \left\lfloor \frac{27}{1} \right\rfloor + 1 = 27 + 1 = 28
$$

**答案**: 输出大小为 **28×28×8**

**解释**：
- 空间维度从 32×32 变为 28×28（由于卷积操作和无填充）
- 通道数从 3（RGB）变为 8（滤波器数量）
- 每个滤波器产生一个输出通道

---

## Question 2
**问题**: 如果填充改为 "same"，输出大小如何变化？

### 解答：

**"same" 填充的含义**：
"same" 填充意味着输出的空间维度与输入相同（当 stride=1 时）。

为了保持输出为 32×32，需要的填充大小计算：

$$
\text{Output Size} = \left\lfloor \frac{\text{Input Size} - \text{Kernel Size} + 2 \times \text{Padding}}{\text{Stride}} \right\rfloor + 1 = \text{Input Size}
$$

对于 kernel size = 5, stride = 1:

$$
32 = \left\lfloor \frac{32 - 5 + 2 \times \text{Padding}}{1} \right\rfloor + 1
$$

$$
31 = 32 - 5 + 2 \times \text{Padding}
$$

$$
2 \times \text{Padding} = 4 \quad \Rightarrow \quad \text{Padding} = 2
$$

**答案**: 输出大小为 **32×32×8**

**解释**：
- "same" 填充在图像周围添加 2 个像素的填充（上下左右各 2 像素）
- 这样可以保持空间维度不变
- 通道数仍然是 8（滤波器数量）

---

## Question 3
**问题**: 如果对 64×64 的输入应用 3×3 的滤波器，步长为 2，无填充，输出的空间大小是多少？

### 解答：

**公式应用**：
- Input Size = 64
- Kernel Size = 3
- Stride = 2
- Padding = 0

$$
\text{Output Size} = \left\lfloor \frac{64 - 3 + 2 \times 0}{2} \right\rfloor + 1 = \left\lfloor \frac{61}{2} \right\rfloor + 1 = 30 + 1 = 31
$$

**答案**: 输出空间大小为 **31×31**

**解释**：
- 步长为 2 意味着滤波器每次移动 2 个像素
- 这大约将空间维度减半
- 输出尺寸 = (64-3)/2 + 1 = 31

---

## Question 4
**问题**: 对 16×16 的特征图应用 2×2 的最大池化层，步长为 2，输出大小是多少？

### 解答：

**公式应用**（最大池化通常无填充）：
- Input Size = 16
- Kernel Size = 2
- Stride = 2
- Padding = 0

$$
\text{Output Size} = \left\lfloor \frac{16 - 2 + 2 \times 0}{2} \right\rfloor + 1 = \left\lfloor \frac{14}{2} \right\rfloor + 1 = 7 + 1 = 8
$$

**答案**: 输出大小为 **8×8**

**解释**：
- 最大池化层将特征图的空间维度减半
- 2×2 池化窗口，步长为 2，正好将 16×16 缩小到 8×8
- 通道数保持不变（池化层不改变通道数）
- 池化层通过选择局部区域的最大值来减少特征图大小

---

## Question 5
**问题**: 一个 128×128 的图像通过两个连续的卷积层。每个层使用 3×3 的卷积核，步长为 1，"same" 填充。输出形状是什么？

### 解答：

**第一个卷积层**：
- Input: 128×128
- Kernel: 3×3, stride=1, padding="same"
- Output: 128×128（"same" 填充保持尺寸不变）

**第二个卷积层**：
- Input: 128×128
- Kernel: 3×3, stride=1, padding="same"
- Output: 128×128（"same" 填充保持尺寸不变）

**答案**: 输出形状为 **128×128**（空间维度）

**解释**：
- "same" 填充确保当 stride=1 时，输出空间维度与输入相同
- 对于 3×3 卷积核，"same" 填充需要 padding=1
- 两个连续的 "same" 卷积层都保持空间维度不变
- 通道数取决于每层的滤波器数量（题目未指定，但空间维度仍为 128×128）

**数学验证**（padding=1）：
$$
\text{Output Size} = \left\lfloor \frac{128 - 3 + 2 \times 1}{1} \right\rfloor + 1 = \left\lfloor \frac{127}{1} \right\rfloor + 1 = 128
$$

---

## Question 6
**问题**: 在课堂示例中，在开始训练循环之前我们运行了 `model.train()`。如果删除这一行会发生什么？

### 解答：

**`model.train()` 的作用**：

`model.train()` 将模型设置为**训练模式**，这对某些层的行为至关重要：

1. **Dropout 层**：
   - 训练模式：随机丢弃一部分神经元（按设定的概率）
   - 评估模式：保留所有神经元，不进行丢弃

2. **Batch Normalization 层**：
   - 训练模式：使用当前批次的均值和方差进行归一化，并更新运行统计量
   - 评估模式：使用训练期间累积的运行均值和方差

3. **其他层**（如 DropConnect, Layer Normalization 等）也可能有不同的行为

**如果删除 `model.train()` 会发生什么？**

**情况分析**：

- **模型刚初始化后**：默认处于训练模式，删除这行可能没有立即影响
- **在调用 `model.eval()` 之后**：如果之前调用了 `model.eval()`（用于验证/测试），然后忘记调用 `model.train()` 就开始训练，会导致：
  
  1. **Dropout 不会激活**：所有神经元都会被保留，导致：
     - 失去正则化效果
     - 可能过拟合
     - 训练和测试之间的行为不一致
  
  2. **Batch Normalization 行为错误**：使用固定的统计量而不是当前批次的统计量，导致：
     - 归一化不正确
     - 梯度计算错误
     - 模型无法正确学习
     - 训练性能显著下降

**答案**：如果删除 `model.train()`，特别是在 `model.eval()` 之后，会导致：
- **Dropout 层不工作**，失去正则化效果
- **Batch Normalization 使用错误的统计量**，导致训练失败
- **模型可能无法正确学习或收敛**
- **训练和推理行为不一致**

**最佳实践**：
```python
# 训练阶段
model.train()  # 设置为训练模式
for epoch in range(num_epochs):
    for batch in train_loader:
        # 训练代码
        ...

# 验证/测试阶段
model.eval()  # 设置为评估模式
with torch.no_grad():  # 禁用梯度计算
    for batch in val_loader:
        # 验证代码
        ...
```

---

## 总结

| 问题 | 答案 | 关键概念 |
|------|------|----------|
| Q1 | 28×28×8 | 基本卷积输出计算 |
| Q2 | 32×32×8 | "same" 填充保持尺寸 |
| Q3 | 31×31 | Stride=2 大约减半尺寸 |
| Q4 | 8×8 | 池化层缩小特征图 |
| Q5 | 128×128 | 连续 "same" 卷积保持尺寸 |
| Q6 | 训练失败 | Dropout/BN 行为模式切换 |

---

## 附录：Python 验证代码

```python
import torch
import torch.nn as nn

# Question 1: 验证输出尺寸
input_q1 = torch.randn(1, 3, 32, 32)
conv_q1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=1, padding=0)
output_q1 = conv_q1(input_q1)
print(f"Q1 Output shape: {output_q1.shape}")  # torch.Size([1, 8, 28, 28])

# Question 2: "same" 填充
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

# Question 5: 两个连续的 "same" 卷积
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

# 训练模式
model.train()
output_train = model(x)
print(f"Q6 Train mode - Dropout active, BN uses batch stats")

# 评估模式
model.eval()
output_eval = model(x)
print(f"Q6 Eval mode - Dropout inactive, BN uses running stats")
print(f"Outputs are different: {not torch.allclose(output_train, output_eval)}")
```

**运行结果**：
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
