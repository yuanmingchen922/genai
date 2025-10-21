#!/bin/bash

# 快速启动脚本 - GenAI API with RNN

echo "================================================"
echo "GenAI API - RNN 文本生成快速启动"
echo "================================================"
echo ""

# 检查是否已安装 Python
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到 Python 3"
    echo "请先安装 Python 3.10 或更高版本"
    exit 1
fi

# 检查是否已安装依赖
echo "📦 检查依赖..."
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "正在安装 Python 依赖..."
    pip install -r requirements.txt
    python3 -m spacy download en_core_web_lg
else
    echo "✅ 依赖已安装"
fi

# 检查是否已有训练好的模型
if [ ! -f "models/rnn_text_generator.pth" ]; then
    echo ""
    echo "⚠️  未找到训练好的 RNN 模型"
    echo ""
    read -p "是否现在训练模型？这将需要几分钟时间 (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "🏋️  开始训练 RNN 模型..."
        python3 -m app.train_rnn
        echo ""
    else
        echo "⚠️  警告: 没有训练模型，RNN 端点将返回未训练的结果"
        echo ""
    fi
else
    echo "✅ 找到已训练的 RNN 模型"
fi

# 启动服务器
echo ""
echo "🚀 启动 FastAPI 服务器..."
echo "服务器地址: http://localhost:8000"
echo "API 文档: http://localhost:8000/docs"
echo ""
echo "按 Ctrl+C 停止服务器"
echo ""

uvicorn app.main:app --reload --port 8000
