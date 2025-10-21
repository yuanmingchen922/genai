# GenAI API - 完整使用指南

## 📚 学习总结

### Module 6 RNN 代码逻辑

#### 1. **LSTM 模型架构** (`LSTMModel`)
```python
class LSTMModel(nn.Module):
    - Embedding 层 (vocab_size → embedding_dim=100)
      将词索引转换为密集向量表示
    
    - LSTM 层 (embedding_dim=100 → hidden_dim=128)
      处理序列，捕获长期依赖关系
    
    - 全连接层 (hidden_dim=128 → vocab_size)
      将 LSTM 输出映射回词汇表大小，用于预测下一个词
```

**关键概念**：
- **Embedding**: 将离散的词索引映射到连续的向量空间
- **LSTM**: 解决 RNN 的梯度消失问题，能记住长期依赖
- **Hidden State**: LSTM 的记忆单元，携带序列的上下文信息

#### 2. **文本预处理流程**
```python
原始文本 → 清理 → 分词 → 构建词汇表 → 编码 → 序列化
```

步骤详解：
1. **清理**: 去除标点、转小写 (`re.sub(r"[^a-zA-Z0-9\s]", "", text)`)
2. **分词**: 按空格分割 (`text.split()`)
3. **词汇表**: 取前 9998 个高频词 + `<PAD>` + `<UNK>`
4. **编码**: 词 → 索引映射
5. **序列化**: 创建固定长度的训练样本 (30 词窗口)

#### 3. **训练过程**
```python
for epoch in range(15):
    for inputs, targets in train_loader:
        # 前向传播
        outputs, _ = model(inputs)
        
        # 计算损失 (交叉熵)
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**关键点**：
- **输入/输出**: 输入是 30 个词，输出是对应的下一个词
- **损失函数**: CrossEntropyLoss，衡量预测分布与真实分布的差异
- **优化器**: Adam，自适应学习率

#### 4. **文本生成过程**
```python
def generate_text(seed_text, length, temperature):
    # 1. 编码种子文本
    input_ids = [vocab.get(w, vocab["<UNK>"]) for w in words]
    
    # 2. 迭代生成
    for _ in range(length):
        # 前向传播
        output, hidden = model(input_tensor, hidden)
        
        # 应用温度参数
        logits = output[0, -1] / temperature
        
        # 采样下一个词
        probs = softmax(logits)
        next_id = multinomial(probs)
        
        # 解码并添加到序列
        words.append(inv_vocab[next_id])
    
    return " ".join(words)
```

**温度参数 (Temperature)**：
- `temperature = 1.0`: 标准采样
- `temperature > 1.0`: 更随机，更有创意
- `temperature < 1.0`: 更确定性，更保守

### 截图中的方法实现

根据截图中的要求，我已经实现了以下内容：

#### 1. **代码更新** ✅
```python
# 原来的 Bigram 模型
bigram_model = BigramModel(corpus)

# 现在替换为 RNN 模型
rnn_generator = RNNTextGenerator()
```

#### 2. **新的文本生成端点** ✅
```python
class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

@app.post("/generate_with_rnn")
def generate_with_rnn(request: TextGenerationRequest):
    generated_text = rnn_generator.generate_text(
        seed_text=request.start_word,
        length=request.length
    )
    return {"generated_text": generated_text}
```

## 🚀 快速开始

### 方法 1: 使用快速启动脚本
```bash
cd /Users/yuanmingchen/Desktop/genai
./start.sh
```

这个脚本会：
1. ✅ 检查 Python 和依赖
2. ✅ 询问是否训练模型（如果未训练）
3. ✅ 启动 FastAPI 服务器

### 方法 2: 手动步骤

#### 步骤 1: 安装依赖
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

#### 步骤 2: 训练 RNN 模型
```bash
python -m app.train_rnn
```

**训练输出示例**：
```
Initializing RNN Text Generator...
Starting training...
Epoch 1/15, Loss: 6.2341
Epoch 2/15, Loss: 5.8123
...
Epoch 15/15, Loss: 3.2156
Training completed!

Testing the trained model...
Seed: 'the count of monte cristo'
Generated: the count of monte cristo was a young sailor who had been...
```

#### 步骤 3: 启动 API
```bash
uvicorn app.main:app --reload --port 8000
```

#### 步骤 4: 测试 API
```bash
python test_api.py
```

### 方法 3: 使用 Docker

#### 构建并运行
```bash
docker-compose up --build
```

#### 后台运行
```bash
docker-compose up -d
```

#### 查看日志
```bash
docker-compose logs -f genai-api
```

## 📝 API 使用示例

### 1. RNN 文本生成

**请求**:
```bash
curl -X POST "http://localhost:8000/generate_with_rnn" \
  -H "Content-Type: application/json" \
  -d '{
    "start_word": "the count of monte cristo",
    "length": 50
  }'
```

**响应**:
```json
{
  "generated_text": "the count of monte cristo was a young sailor who...",
  "start_word": "the count of monte cristo",
  "length": 50,
  "model": "LSTM"
}
```

### 2. 比较两种模型

**Bigram (简单)**:
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"start_word": "the", "length": 20}'
```

**RNN/LSTM (高级)**:
```bash
curl -X POST "http://localhost:8000/generate_with_rnn" \
  -H "Content-Type: application/json" \
  -d '{"start_word": "the", "length": 20}'
```

### 3. 其他功能

**词嵌入**:
```bash
curl -X POST "http://localhost:8000/embedding" \
  -H "Content-Type: application/json" \
  -d '{"word": "king", "return_size": 10}'
```

**词相似度**:
```bash
curl -X POST "http://localhost:8000/similarity" \
  -H "Content-Type: application/json" \
  -d '{"word1": "king", "word2": "queen"}'
```

## 🧪 测试验证

### 运行完整测试套件
```bash
python test_api.py
```

**测试内容**:
- ✅ 根端点
- ✅ Bigram 文本生成
- ✅ RNN 文本生成
- ✅ 多种子词 RNN 生成
- ✅ 词嵌入
- ✅ 词相似度

## 📊 项目文件说明

### 新增文件
```
✨ app/rnn_model.py           - RNN 模型实现
✨ app/train_rnn.py           - 训练脚本
✨ Dockerfile                 - Docker 配置
✨ docker-compose.yml         - Docker Compose 配置
✨ requirements.txt           - Python 依赖
✨ test_api.py                - API 测试脚本
✨ start.sh                   - 快速启动脚本
✨ README_RNN.md              - RNN 功能文档
✨ USAGE_GUIDE.md             - 本使用指南
```

### 修改文件
```
🔧 app/main.py                - 添加 RNN 端点
```

## 🔧 高级配置

### 自定义训练参数

编辑 `app/train_rnn.py`:
```python
# 调整训练轮数
generator.train_from_text(epochs=20)  # 默认 15

# 使用自定义文本
with open('my_text.txt', 'r') as f:
    custom_text = f.read()
generator.train_from_text(text_content=custom_text, epochs=10)
```

### 调整模型参数

编辑 `app/rnn_model.py`:
```python
# 修改模型架构
model = LSTMModel(
    vocab_size=10000,      # 词汇表大小
    embedding_dim=100,     # 嵌入维度
    hidden_dim=128         # 隐藏层维度
)

# 修改序列长度
self.seq_len = 50  # 默认 30
```

### 调整生成参数

在 API 调用中：
```python
# 在 app/main.py 中修改
generated_text = rnn_generator.generate_text(
    seed_text=request.start_word,
    length=request.length,
    temperature=0.8  # 调整创意度
)
```

## 🐛 常见问题

### Q1: "Model not trained" 错误
**A**: 运行 `python -m app.train_rnn` 训练模型

### Q2: 内存不足
**A**: 减少 batch_size 或 hidden_dim
```python
# 在 app/rnn_model.py
train_loader = DataLoader(train_dataset, batch_size=32)  # 从 64 降低
model = LSTMModel(hidden_dim=64)  # 从 128 降低
```

### Q3: 生成文本质量差
**A**: 
- 增加训练轮数 (`epochs=20`)
- 使用更大的训练数据
- 调整温度参数 (`temperature=0.7`)

### Q4: API 启动慢
**A**: 这是正常的，因为要加载 spaCy 模型和 RNN 模型

## 📈 性能对比

| 模型 | 复杂度 | 训练时间 | 生成质量 | 上下文理解 |
|------|--------|----------|----------|------------|
| Bigram | 低 | 秒级 | 低 | 仅 1 词 |
| LSTM | 高 | 分钟级 | 高 | 30+ 词 |

## 🎯 学习要点

1. **RNN vs Bigram**: RNN 能捕获长期依赖，生成更连贯的文本
2. **LSTM 优势**: 解决梯度消失，适合长序列
3. **温度采样**: 控制生成文本的随机性和创意性
4. **词嵌入**: 将词映射到连续空间，捕获语义关系
5. **序列建模**: 将文本转换为固定长度的序列进行训练

## 🌟 下一步

- [ ] 尝试不同的温度参数
- [ ] 使用自己的文本数据训练
- [ ] 调整模型架构参数
- [ ] 实现 GRU 模型进行对比
- [ ] 添加文本质量评估指标

## 📚 参考资料

- PyTorch LSTM 文档: https://pytorch.org/docs/stable/nn.html#lstm
- FastAPI 文档: https://fastapi.tiangolo.com/
- Module 6 Practical RNN 代码

---

**完成日期**: 2025年10月20日
**项目状态**: ✅ 配置完成，可以使用
