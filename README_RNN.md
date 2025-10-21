# GenAI API - RNN Text Generation

这个项目是 Module 3 到 Module 7 的集成，将 RNN 文本生成功能添加到 FastAPI Docker 实现中。

## 项目结构

```
genai/
├── app/
│   ├── main.py                 # FastAPI 主应用 (已更新)
│   ├── bigram_model.py         # Bigram 模型 (Module 3)
│   ├── rnn_model.py            # RNN/LSTM 模型 (新增 - Module 7)
│   ├── cnn_classifier.py       # CNN 图像分类器
│   └── train_rnn.py            # RNN 模型训练脚本
├── models/
│   ├── cnn_classifier.pth      # CNN 模型权重
│   ├── rnn_text_generator.pth  # RNN 模型权重 (训练后生成)
│   └── rnn_vocab.pkl           # RNN 词汇表 (训练后生成)
├── Dockerfile                   # Docker 配置文件
├── docker-compose.yml           # Docker Compose 配置
├── requirements.txt             # Python 依赖
└── README_RNN.md               # 本文件
```

## 主要功能

### 1. 文本生成 API
- **Bigram 模型** (`/generate`): 基于简单的二元语法模型
- **RNN/LSTM 模型** (`/generate_with_rnn`): 基于深度学习的文本生成 ✨ 新增

### 2. 词嵌入功能
- `/embedding`: 获取词向量
- `/similarity`: 计算词相似度
- `/sentence-similarity`: 计算句子相似度

### 3. 图像分类
- `/classify-image`: Base64 编码的图像分类
- `/classify-image-file`: 上传文件进行分类

## 使用方法

### 选项 1: 本地运行 (不使用 Docker)

#### 步骤 1: 安装依赖
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

#### 步骤 2: 训练 RNN 模型
```bash
cd /Users/yuanmingchen/Desktop/genai
python -m app.train_rnn
```

这将：
- 下载《基督山伯爵》文本
- 训练 LSTM 模型（15 个 epoch）
- 保存模型到 `models/rnn_text_generator.pth`
- 保存词汇表到 `models/rnn_vocab.pkl`

#### 步骤 3: 启动 API 服务器
```bash
uvicorn app.main:app --reload --port 8000
```

### 选项 2: 使用 Docker (推荐)

#### 步骤 1: 构建 Docker 镜像
```bash
docker-compose build
```

#### 步骤 2: 运行容器
```bash
docker-compose up
```

或者在后台运行：
```bash
docker-compose up -d
```

#### 查看日志
```bash
docker-compose logs -f
```

#### 停止服务
```bash
docker-compose down
```

## API 测试示例

### 1. 测试 RNN 文本生成

```bash
curl -X POST "http://localhost:8000/generate_with_rnn" \
  -H "Content-Type: application/json" \
  -d '{
    "start_word": "the count of monte cristo",
    "length": 50
  }'
```

预期响应：
```json
{
  "generated_text": "the count of monte cristo was a young sailor...",
  "start_word": "the count of monte cristo",
  "length": 50,
  "model": "LSTM"
}
```

### 2. 比较 Bigram vs RNN

**Bigram 模型:**
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "start_word": "the",
    "length": 20
  }'
```

**RNN 模型:**
```bash
curl -X POST "http://localhost:8000/generate_with_rnn" \
  -H "Content-Type: application/json" \
  -d '{
    "start_word": "the",
    "length": 20
  }'
```

### 3. 查看所有端点
访问 API 文档：
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

或者：
```bash
curl http://localhost:8000/
```

## 代码逻辑说明

### RNN 模型架构 (`app/rnn_model.py`)

#### 1. LSTMModel 类
```python
class LSTMModel(nn.Module):
    - Embedding层: 将词索引转换为密集向量
    - LSTM层: 处理序列数据，捕获上下文
    - 全连接层: 将 LSTM 输出映射到词汇表大小
```

#### 2. RNNTextGenerator 类
主要方法：
- `train_from_text()`: 从文本训练模型
  - 文本预处理（清理、分词）
  - 构建词汇表（前 10000 个高频词）
  - 创建训练序列（30 词窗口）
  - LSTM 训练循环
  
- `generate_text()`: 生成文本
  - 将种子文本编码为词索引
  - 迭代预测下一个词
  - 使用温度参数控制随机性
  - 解码为可读文本

- `save_model()` / `load_model()`: 模型持久化

### 与 Module 6 代码的对应关系

| Module 6 代码 | app/rnn_model.py | 说明 |
|--------------|------------------|------|
| `LSTMModel` 类 | `LSTMModel` 类 | 相同的 LSTM 架构 |
| `TextDataset` 类 | `TextDataset` 类 | 序列数据集 |
| `generate_text()` 函数 | `generate_text()` 方法 | 文本生成逻辑 |
| 训练循环 | `train_from_text()` 方法 | 封装的训练过程 |
| `vocab` 字典 | `self.vocab` | 词汇表映射 |

### FastAPI 集成 (`app/main.py`)

新增的端点：
```python
@app.post("/generate_with_rnn")
async def generate_with_rnn(request: TextGenerationRequest):
    # 使用全局 rnn_generator 实例
    # 调用 generate_text() 方法
    # 返回生成的文本
```

## 技术栈

- **FastAPI**: Web 框架
- **PyTorch**: 深度学习框架
- **LSTM**: 长短期记忆网络（文本生成）
- **spaCy**: 词嵌入
- **Docker**: 容器化部署

## 注意事项

1. **模型训练时间**: 首次训练需要几分钟，取决于硬件性能
2. **内存要求**: 训练和推理都需要适量内存（推荐至少 4GB RAM）
3. **预训练模型**: 如果已有训练好的模型，将其放在 `models/` 目录下
4. **文本质量**: RNN 生成的文本质量取决于训练数据和训练时长

## 故障排查

### 问题: "Model not trained" 错误
**解决**: 运行 `python -m app.train_rnn` 训练模型

### 问题: Docker 构建失败
**解决**: 确保有足够的磁盘空间，检查网络连接

### 问题: spaCy 模型未找到
**解决**: 
```bash
python -m spacy download en_core_web_lg
```

## 下一步改进

- [ ] 添加更多训练参数的 API 配置
- [ ] 支持自定义训练数据上传
- [ ] 实现模型版本管理
- [ ] 添加生成文本的质量评估
- [ ] 支持 GRU 等其他 RNN 变体

## 许可证

本项目用于教育目的。
