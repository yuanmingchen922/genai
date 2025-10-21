#!/usr/bin/env python3
"""
快速验证配置是否正确
"""
import os
import sys

def check_file(path, required=True):
    """检查文件是否存在"""
    exists = os.path.exists(path)
    status = "✅" if exists else ("❌" if required else "⚠️ ")
    print(f"{status} {path}")
    return exists

def check_python_package(package_name):
    """检查 Python 包是否已安装"""
    try:
        __import__(package_name)
        print(f"✅ {package_name}")
        return True
    except ImportError:
        print(f"❌ {package_name}")
        return False

def main():
    print("=" * 60)
    print("GenAI RNN 配置验证")
    print("=" * 60)
    
    base_dir = "/Users/yuanmingchen/Desktop/genai"
    os.chdir(base_dir)
    
    # 检查核心文件
    print("\n📁 核心文件:")
    files_ok = True
    files_ok &= check_file("app/main.py")
    files_ok &= check_file("app/bigram_model.py")
    files_ok &= check_file("app/rnn_model.py")
    files_ok &= check_file("app/cnn_classifier.py")
    files_ok &= check_file("app/train_rnn.py")
    
    # 检查配置文件
    print("\n📄 配置文件:")
    files_ok &= check_file("Dockerfile")
    files_ok &= check_file("docker-compose.yml")
    files_ok &= check_file("requirements.txt")
    files_ok &= check_file(".dockerignore")
    
    # 检查脚本
    print("\n🔧 脚本文件:")
    files_ok &= check_file("start.sh")
    files_ok &= check_file("test_api.py")
    
    # 检查文档
    print("\n📚 文档文件:")
    check_file("README_RNN.md")
    check_file("USAGE_GUIDE.md")
    check_file("LEARNING_SUMMARY.md")
    
    # 检查模型目录
    print("\n📦 模型文件 (可选):")
    check_file("models/cnn_classifier.pth", required=False)
    check_file("models/rnn_text_generator.pth", required=False)
    check_file("models/rnn_vocab.pkl", required=False)
    
    # 检查 Python 依赖
    print("\n🐍 Python 依赖:")
    packages = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "torch",
        "torchvision",
        "spacy",
        "numpy",
        "PIL",
        "requests"
    ]
    
    packages_ok = True
    for pkg in packages:
        packages_ok &= check_python_package(pkg)
    
    # 检查 spaCy 模型
    print("\n🔤 spaCy 模型:")
    try:
        import spacy
        nlp = spacy.load("en_core_web_lg")
        print("✅ en_core_web_lg")
        spacy_ok = True
    except:
        print("❌ en_core_web_lg (运行: python -m spacy download en_core_web_lg)")
        spacy_ok = False
    
    # 总结
    print("\n" + "=" * 60)
    print("验证总结")
    print("=" * 60)
    
    all_ok = files_ok and packages_ok and spacy_ok
    
    if all_ok:
        print("🎉 所有配置正确！")
        print("\n下一步:")
        print("1. 训练模型: python -m app.train_rnn")
        print("2. 启动服务: ./start.sh")
        print("   或: uvicorn app.main:app --reload")
        print("3. 测试 API: python test_api.py")
        return 0
    else:
        print("⚠️  存在缺失的依赖")
        print("\n修复步骤:")
        if not packages_ok:
            print("1. 安装 Python 依赖: pip install -r requirements.txt")
        if not spacy_ok:
            print("2. 下载 spaCy 模型: python -m spacy download en_core_web_lg")
        return 1

if __name__ == "__main__":
    sys.exit(main())
