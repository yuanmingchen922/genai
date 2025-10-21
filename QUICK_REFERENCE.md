# 🚀 快速参考卡

## 📂 项目文件导航

### 🆕 新增文件 (15 个)

#### 核心代码 (5 个)
```
✨ app/rnn_model.py          - RNN 模型实现 (237 行)
✨ app/train_rnn.py          - 训练脚本 (57 行)
✨ test_api.py               - API 测试 (146 行)
✨ verify_setup.py           - 配置验证 (92 行)
✨ start.sh                  - 快速启动脚本 (44 行)
```

#### Docker 配置 (3 个)
```
✨ Dockerfile                - Docker 镜像配置
✨ docker-compose.yml        - 容器编排
✨ .dockerignore             - 构建优化
```

#### Python 依赖 (1 个)
```
✨ requirements.txt          - 所有依赖包
```

#### 文档文件 (6 个)
```
✨ README_RNN.md            - RNN 功能说明
✨ USAGE_GUIDE.md           - 完整使用指南
✨ LEARNING_SUMMARY.md      - 学习总结
✨ ARCHITECTURE.md          - 架构图示
✨ SETUP_COMPLETE.md        - 完成总结
✨ PROJECT_REPORT.md        - 项目报告
✨ QUICK_REFERENCE.md       - 本快速参考
```

### 🔧 修改文件 (1 个)
```
🔧 app/main.py              - 添加 RNN 端点
```

---

## ⚡ 快速命令

### 一键启动
```bash
./start.sh
```

### 验证配置
```bash
python verify_setup.py
```

### 训练模型
```bash
python -m app.train_rnn
```

### 启动服务
```bash
# 方式 1: 开发模式
uvicorn app.main:app --reload

# 方式 2: Docker
docker-compose up

# 方式 3: 后台运行
docker-compose up -d
```

### 测试 API
```bash
python test_api.py
```

### RNN 生成
```bash
curl -X POST http://localhost:8000/generate_with_rnn \
  -H "Content-Type: application/json" \
  -d '{"start_word": "the count", "length": 30}'
```

---

## 📖 文档阅读顺序

### 🎓 学习路径
```
1️⃣ LEARNING_SUMMARY.md     → 理解 RNN 原理
2️⃣ ARCHITECTURE.md          → 查看系统架构
3️⃣ USAGE_GUIDE.md           → 学习使用方法
4️⃣ README_RNN.md            → 了解功能特性
5️⃣ PROJECT_REPORT.md        → 查看完成情况
```

### 🛠️ 使用路径
```
1️⃣ SETUP_COMPLETE.md        → 快速上手
2️⃣ QUICK_REFERENCE.md       → 命令参考 (本文档)
3️⃣ verify_setup.py          → 验证环境
4️⃣ start.sh                 → 一键启动
5️⃣ test_api.py              → 测试验证
```

---

## 🎯 核心 API 端点

### 文本生成
```bash
# Bigram 模型
POST /generate
{
  "start_word": "the",
  "length": 20
}

# RNN 模型 ⭐
POST /generate_with_rnn
{
  "start_word": "the count of monte cristo",
  "length": 50
}
```

### 词嵌入
```bash
POST /embedding
{
  "word": "king",
  "return_size": 10
}

POST /similarity
{
  "word1": "king",
  "word2": "queen"
}
```

### 图像分类
```bash
POST /classify-image
{
  "image_data": "base64...",
  "top_k": 3
}
```

---

## 🔧 常用操作

### Docker 相关
```bash
# 构建镜像
docker-compose build

# 启动服务
docker-compose up

# 后台运行
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down

# 重建并启动
docker-compose up --build
```

### Python 相关
```bash
# 安装依赖
pip install -r requirements.txt

# 下载 spaCy 模型
python -m spacy download en_core_web_lg

# 训练 RNN
python -m app.train_rnn

# 启动 API
uvicorn app.main:app --reload --port 8000

# 运行测试
python test_api.py

# 验证配置
python verify_setup.py
```

---

## 📊 项目统计

### 代码量
- **总代码行数**: 657+ 行
- **新增文件数**: 15 个
- **修改文件数**: 1 个
- **文档页数**: 约 50+ 页

### 功能统计
- **支持模型**: 3 个 (Bigram, RNN, CNN)
- **API 端点**: 7 个
- **部署方式**: 3 种
- **测试函数**: 6 个

---

## 🎓 学习重点

### LSTM 核心
```
Embedding → LSTM → Linear
   ↓         ↓       ↓
 词向量   隐藏状态  预测
```

### 训练流程
```
文本 → 清理 → 分词 → 词汇表 → 编码 → 训练 → 保存
```

### 生成流程
```
种子 → 编码 → LSTM → 采样 → 解码 → 输出
```

---

## 🌐 访问地址

### 本地开发
- API: http://localhost:8000
- Swagger: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Docker 部署
- API: http://localhost:8000
- 容器名: genai-api
- 镜像: genai-api:latest

---

## 🐛 故障排查

### 问题 1: 依赖缺失
```bash
# 解决方案
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

### 问题 2: 模型未训练
```bash
# 解决方案
python -m app.train_rnn
```

### 问题 3: 端口占用
```bash
# 检查端口
lsof -i :8000

# 使用其他端口
uvicorn app.main:app --port 8080
```

### 问题 4: Docker 构建失败
```bash
# 清理缓存
docker-compose down
docker system prune -a

# 重新构建
docker-compose build --no-cache
```

---

## 💡 最佳实践

### 开发流程
1. ✅ 验证配置: `python verify_setup.py`
2. ✅ 训练模型: `python -m app.train_rnn`
3. ✅ 启动服务: `./start.sh`
4. ✅ 测试 API: `python test_api.py`
5. ✅ 查看文档: http://localhost:8000/docs

### 生产部署
1. ✅ 使用 Docker: `docker-compose up -d`
2. ✅ 配置环境变量
3. ✅ 设置日志记录
4. ✅ 监控服务状态
5. ✅ 定期备份模型

---

## 📞 快速帮助

### 需要了解原理？
→ 阅读 `LEARNING_SUMMARY.md`

### 需要开始使用？
→ 运行 `./start.sh`

### 需要测试功能？
→ 运行 `python test_api.py`

### 需要部署上线？
→ 运行 `docker-compose up -d`

### 需要查看架构？
→ 阅读 `ARCHITECTURE.md`

### 需要配置帮助？
→ 阅读 `USAGE_GUIDE.md`

---

## 🎯 核心文件速查

| 文件 | 用途 | 何时使用 |
|------|------|----------|
| `start.sh` | 快速启动 | 每次启动服务 |
| `verify_setup.py` | 验证配置 | 初次使用/排查问题 |
| `test_api.py` | 测试 API | 验证功能是否正常 |
| `app/train_rnn.py` | 训练模型 | 首次使用/重新训练 |
| `app/rnn_model.py` | RNN 实现 | 理解模型/修改功能 |
| `app/main.py` | API 主文件 | 添加/修改端点 |
| `requirements.txt` | 依赖列表 | 安装环境 |
| `Dockerfile` | Docker 配置 | 容器化部署 |

---

## ✅ 检查清单

### 首次使用
- [ ] 阅读 `SETUP_COMPLETE.md`
- [ ] 运行 `python verify_setup.py`
- [ ] 安装依赖 `pip install -r requirements.txt`
- [ ] 下载 spaCy `python -m spacy download en_core_web_lg`
- [ ] 训练模型 `python -m app.train_rnn`
- [ ] 启动服务 `./start.sh`
- [ ] 测试 API `python test_api.py`

### 日常使用
- [ ] 启动服务 `./start.sh`
- [ ] 访问文档 http://localhost:8000/docs
- [ ] 测试端点

### 部署生产
- [ ] 构建 Docker `docker-compose build`
- [ ] 启动容器 `docker-compose up -d`
- [ ] 查看日志 `docker-compose logs -f`
- [ ] 测试连接

---

**创建时间**: 2025年10月20日
**版本**: 1.0
**状态**: ✅ 完成

需要更多帮助？查看完整文档！
