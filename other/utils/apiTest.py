import requests
import json

url = "http://localhost:8012/v1/chat/completions"
headers = {"Content-Type": "application/json"}

# 1、测试全局搜索  graphrag-global-search:latest
global_data = {
    "model": "graphrag-global-search:latest",
    "messages": [{"role": "user", "content": "这个故事的首要主题是什么?记住请使用中文进行回答，不要用英文。"}],
    "temperature": 0.7,
    # "stream": True,#True or False
}

# 2、测试本地搜索  graphrag-local-search:latest
local_data = {
    "model": "graphrag-local-search:latest",
    "messages": [{"role": "user", "content": "唐僧是谁，他的主要关系是什么?记住请使用中文进行回答，不要用英文。"}],
    "temperature": 0.7,
    # "stream": True,#True or False
}

# 3、测试全局和本地搜索  full-model:latest
full_data = {
    "model": "full-model:latest",
    "messages": [{"role": "user", "content": "唐僧是谁，他的主要关系是什么?记住请使用中文进行回答，不要用英文。"}],
    "temperature": 0.7,
    # "stream": True,#True or False
}

# 接收非流式输出
# 1、测试全局搜索  graphrag-global-search:latest
response = requests.post(url, headers=headers, data=json.dumps(global_data))
# 2、测试本地搜索  graphrag-local-search:latest
# response = requests.post(url, headers=headers, data=json.dumps(local_data))
# 3、测试全局和本地搜索  full-model:latest
# response = requests.post(url, headers=headers, data=json.dumps(full_data))

# print(response.json())
print(response.json()['choices'][0]['message']['content'])





# # 接收非流式输出
# try:
#     with requests.post(url, stream=True, headers=headers, data=json.dumps(data)) as response:
#         for line in response.iter_lines():
#         # for line in response.iter_content(chunk_size=16):
#             if line:
#                 json_str = line.decode('utf-8').strip("data: ")
#
#                 # 检查是否为空或不合法的字符串
#                 if not json_str:
#                     print("Received empty string, skipping...")
#                     continue
#
#                 # 确保字符串是有效的JSON格式
#                 if json_str.startswith('{') and json_str.endswith('}'):
#                     try:
#                         data = json.loads(json_str)
#                         print(f"Received JSON data: {data['choices'][0]['delta']['content']}")
#                         # print(f"{data['choices'][0]['delta']['content']}")
#                     except json.JSONDecodeError as e:
#                         print(f"Failed to decode JSON: {e}")
#                 else:
#                     print(f"Invalid JSON format: {json_str}")
# except Exception as e:
#     print(f"Error occurred: {e}")





