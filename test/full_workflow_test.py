import os
import requests
import json
import time
import sys
import logging
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection
from langchain_openai import ChatOpenAI

load_dotenv()

def connect_to_milvus():
    try:
        connections.connect(
            alias="default",
            host='localhost',
            port=19530
        )
        logging.info("成功连接至 Milvus")
    except Exception as e:
        logging.error(f"连接 Milvus 失败: {e}")
        exit(1)

def search_collection(collection, query_vector):
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10}
    }
    search_results = collection.search(
        data=query_vector,
        anns_field="vectors",
        param=search_params,
        limit=5,
        output_fields=["metadata"]
    )
    return search_results

def milvus_answer(query_list):
    model = SentenceTransformer(r"D:\code\rag-retrieval-main\embeding\bge-large-zh-v1.5\BAAI\bge-large-zh-v1___5",
                                device="cuda")
    connect_to_milvus()
    collection_name = "csi_name"
    collection = Collection(collection_name)

    context=[]
    for query in query_list:
        query_vector = [model.encode(str(query)).tolist()]

        retriever = search_collection(collection, query_vector)
        # print(retriever)
        retriever = str(retriever)
        print('INFO:retriever:', retriever)
        template = """你是一个数据分类助手，利用上下文的已分类数据，只关注上下文中'input'和'output'的内容，对用户输入的未分类数据进行分类，只需要将'output'的内容作为回答即可，不能有多余的请求和回答。
            上下文:{context}
            用户输入:{user_input}
            answer:
            """
        prompt = template.format(context=retriever, user_input=query)
        # print(prompt)
        # prompt = ChatPromptTemplate.from_template(template)
        llm = ChatOpenAI(model="gpt-3.5-turbo",
                         base_url=os.getenv('OPENAI_BASE_URL', 'https://xiaoai.plus/v1'),
                         api_key=os.getenv('OPENAI_API_KEY'))

        answer = llm.invoke(prompt)
        #print('INFO:answer.content:', answer.content)
        context.append(answer.content)
    return context


#因使用次数限制，请到coze平台注册独立的api账号，在线数据库发布到api即可
def coze_answer(query):
    workflow_id = os.getenv('COZE_WORKFLOW_ID')
    app_id = os.getenv('COZE_APP_ID')
    access_token = os.getenv('COZE_ACCESS_TOKEN')
    # user_id = "你的USER-ID"

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
        "auto_save_history": not (True),
        "additional_messages": [
            {
                "role": "user",
                "content_type": "text",
                # "content": "在房间C601，2号工位的人是谁",
                "content": query
            }
        ],
        "parameters": {
            "input": "123"
        }
    }
    response = requests.post(api_url, headers=headers, json=body)
    # print(response.text)
    data = []
    for line in response.iter_lines():
        decoded_line = line.decode('utf-8', errors='ignore')  # 解码
        # print(decoded_line)
        if decoded_line.startswith("event:"):  # 标记event
            event = decoded_line[6:]
            # print(event)
        if decoded_line.startswith("data:"):
            event_data = json.loads(decoded_line[5:])
            data.append(event_data)
            # print(event_data)
            if event == 'conversation.message.delta':  # 流式输出标记
                sys.stdout.write(event_data["content"])

                time.sleep(0.1)

    # print(data)

    data = data[-4]['content']
    data = json.loads(data)

    if data["nottimes"]:
        identity_list_str = data["nottimes"]
    else:
        identity_list_str = data["realtime"]
    #print('INFO:identity_list_str:',identity_list_str)

    return identity_list_str

def final_answer(query,context):
    # query='早上10点张三在办公室吗'
    # query='早上10点李四在办公室吗'
    template = """
        在固定场景下：所有人都是固定工位，已知信息都是依据对应问题从对应位置采集的数据，请按流程回答问题。
        1.理解问题后，根据已知信息，生成一个简短的句子，描述与问题相关的情境。
        2.根据重新生成的信息和问题情境，直接回答查询中的问题。
        3.如果没有足够信息回答问题，请直接说明不知道。
        示例：
        问题："早上10点李四在办公室吗"
        已知信息："张三"
        代表从李四原本所在的位置采集的信息是张三，信息不匹配，则意味着李四不在，而张三在办公室
        重生成内容：李四不在，而张三在办公室
        回答：李四不在办公室
        
        start：
        问题：{query}
        已知信息：{context}
    """
    prompt = template.format(query=query, context=context)
    # print(prompt)
    # prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-3.5-turbo",
                     base_url='https://xiaoai.plus/v1',
                     api_key=os.getenv('OPENAI_API_KEY'))

    answer = llm.invoke(prompt)
    answer = answer.content
    #print('INFO:ANSWER:',answer)
    data = answer.split("回答：")[1].strip()
    #print('INFO:DATA:',data)
    return data

def chat(query):


    query_list=coze_answer(query)
    #print('INFO:identity_list_type:', type(query_list))
    context=milvus_answer(query_list)
    answer=final_answer(query,context)
    #print('INFO:answer:',answer)
    return answer

def main():
    query = "C601房间2号工位的人是谁"
    answer=chat(query)
    print(answer)
main()
