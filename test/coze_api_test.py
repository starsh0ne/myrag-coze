import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

def test_coze_api():
    workflow_id = os.getenv('COZE_WORKFLOW_ID')
    app_id = os.getenv('COZE_APP_ID')
    access_token = os.getenv('COZE_ACCESS_TOKEN')

    api_url = 'https://api.coze.cn/v1/workflows/chat'
    headers = {
        'Authorization': access_token,
        'Content-Type': 'application/json'
    }
    body = {
        "workflow_id": workflow_id,
        "app_id": app_id,
        "user_id": "123123",
        "stream": True,
        "auto_save_history": False,
        "additional_messages": [
            {
                "role": "user",
                "content_type": "text",
                "content": "2025年3月18号10点累成套在不在实验室"
            }
        ],
        "parameters": {
            "input": "123"
        }
    }

    print("发送请求到:", api_url)
    print("请求头:", headers)
    print("请求体:", json.dumps(body, ensure_ascii=False, indent=2))

    try:
        response = requests.post(api_url, headers=headers, json=body, timeout=30)
        print(f"\n响应状态码: {response.status_code}")
        print(f"响应头: {dict(response.headers)}")
        print(f"响应内容: {response.text}")

        if response.status_code == 200:
            print("\n成功！解析流式响应:")
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    print(line_str)
                    if line_str.startswith("data:"):
                        try:
                            data = json.loads(line_str[5:])
                            print(f"  解析结果: {data}")
                        except:
                            pass
        else:
            print(f"\n失败！状态码: {response.status_code}")

    except Exception as e:
        print(f"\n错误: {e}")


if __name__ == "__main__":
    test_coze_api()