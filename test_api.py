"""
测试脚本 - 验证 RNN 文本生成 API
运行前确保 API 服务器正在运行 (http://localhost:8000)
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_root():
    """测试根端点"""
    print("\n" + "="*60)
    print("测试 1: 根端点")
    print("="*60)
    response = requests.get(f"{BASE_URL}/")
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    return response.status_code == 200

def test_bigram_generation():
    """测试 Bigram 文本生成"""
    print("\n" + "="*60)
    print("测试 2: Bigram 文本生成")
    print("="*60)
    data = {
        "start_word": "the",
        "length": 20
    }
    response = requests.post(f"{BASE_URL}/generate", json=data)
    print(f"状态码: {response.status_code}")
    print(f"请求: {json.dumps(data, indent=2)}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    return response.status_code == 200

def test_rnn_generation():
    """测试 RNN 文本生成"""
    print("\n" + "="*60)
    print("测试 3: RNN 文本生成")
    print("="*60)
    data = {
        "start_word": "the count of monte cristo",
        "length": 50
    }
    response = requests.post(f"{BASE_URL}/generate_with_rnn", json=data)
    print(f"状态码: {response.status_code}")
    print(f"请求: {json.dumps(data, indent=2, ensure_ascii=False)}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    return response.status_code == 200

def test_rnn_generation_multiple_seeds():
    """测试多个种子词的 RNN 生成"""
    print("\n" + "="*60)
    print("测试 4: 多个种子词的 RNN 生成")
    print("="*60)
    
    seeds = [
        "once upon a time",
        "the quick brown",
        "in the beginning"
    ]
    
    for seed in seeds:
        print(f"\n--- 种子词: '{seed}' ---")
        data = {
            "start_word": seed,
            "length": 30
        }
        response = requests.post(f"{BASE_URL}/generate_with_rnn", json=data)
        if response.status_code == 200:
            result = response.json()
            print(f"生成文本: {result['generated_text']}")
        else:
            print(f"错误: {response.status_code}")
            print(response.text)
    
    return True

def test_word_embedding():
    """测试词嵌入"""
    print("\n" + "="*60)
    print("测试 5: 词嵌入")
    print("="*60)
    data = {
        "word": "king",
        "return_size": 10
    }
    response = requests.post(f"{BASE_URL}/embedding", json=data)
    print(f"状态码: {response.status_code}")
    print(f"请求: {json.dumps(data, indent=2)}")
    if response.status_code == 200:
        result = response.json()
        print(f"词: {result['word']}")
        print(f"嵌入维度: {result['dimensions_returned']}")
        print(f"嵌入向量 (前10维): {result['embedding'][:10]}")
    return response.status_code == 200

def test_word_similarity():
    """测试词相似度"""
    print("\n" + "="*60)
    print("测试 6: 词相似度")
    print("="*60)
    data = {
        "word1": "king",
        "word2": "queen"
    }
    response = requests.post(f"{BASE_URL}/similarity", json=data)
    print(f"状态码: {response.status_code}")
    print(f"请求: {json.dumps(data, indent=2)}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    return response.status_code == 200

def main():
    """运行所有测试"""
    print("\n" + "🚀 " * 20)
    print("开始测试 GenAI API")
    print("🚀 " * 20)
    
    tests = [
        ("根端点", test_root),
        ("Bigram 生成", test_bigram_generation),
        ("RNN 生成", test_rnn_generation),
        ("RNN 多种子词", test_rnn_generation_multiple_seeds),
        ("词嵌入", test_word_embedding),
        ("词相似度", test_word_similarity),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except requests.exceptions.ConnectionError:
            print(f"\n❌ 无法连接到服务器 {BASE_URL}")
            print("请确保 API 服务器正在运行:")
            print("  本地运行: uvicorn app.main:app --reload")
            print("  Docker: docker-compose up")
            return
        except Exception as e:
            print(f"\n❌ 测试 '{name}' 失败: {str(e)}")
            results.append((name, False))
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    for name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{status} - {name}")
    
    total = len(results)
    passed = sum(1 for _, success in results if success)
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！")
    else:
        print(f"\n⚠️  {total - passed} 个测试失败")

if __name__ == "__main__":
    main()
