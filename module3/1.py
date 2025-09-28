from openai import OpenAI
import os

# 检查是否设置了 API 密钥
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("错误: 请设置 OPENAI_API_KEY 环境变量")
    print("你可以通过以下方式设置:")
    print("export OPENAI_API_KEY='your-api-key-here'")
    exit(1)

client = OpenAI()

try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Write a short bedtime story about a unicorn."}
        ]
    )
    
    print(response.choices[0].message.content)
except Exception as e:
    print(f"调用 OpenAI API 时出错: {e}")
