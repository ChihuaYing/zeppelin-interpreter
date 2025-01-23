import json
import requests

API_KEY = "204a3ea9bf39f18dd9bf32c71ecbb607.mITgz6pgV7Hzj27A"
API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
MODEL = "GLM-4-Flash"

def get_response(prompt):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    request_body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post(API_URL, headers=headers, data=json.dumps(request_body))
    response.raise_for_status()
    response_data = response.json()
    choices = response_data.get("choices", [])
    if not choices:
        return "Error: Missing or empty 'choices' in response"
    return choices[0].get("message", {}).get("content", "")

class UDFAskBigModel:
    def __init__(self):
        pass

    def transform(self, data, args, kvargs):
        print("enter UDFAskBigModel success")
        prompt = kvargs["prompt"].decode("utf-8")
        response = get_response(prompt)

        return [
            ["(response)"],
            ["BINARY"],
            [response.encode("utf-8")]
        ]

if __name__ == '__main__':
    # 本地测试
    udf = UDFAskBigModel()
    data = None
    args = None
    kvargs = {
        "prompt": "你是一个概括大师，我将给你几个用‘;’分隔的中文短语，请你将它们概括成一个中文短语。注意仅需返回概括结果。需要概括的中文短语是: 中国;美国;日本".encode("utf-8")
    }
    result = udf.transform(data, args, kvargs)
    print(result[2][0].decode("utf-8"))

    kvargs = {
        "prompt": "你是一个概念大师，请你用短语给出“中国”和“美国”之间切实具体、简洁精炼的关系，最好不超过10个字，注意仅需返回概括结果".encode("utf-8")
    }
    result = udf.transform(data, args, kvargs)
    print(result[2][0].decode("utf-8"))