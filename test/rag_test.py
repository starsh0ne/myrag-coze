import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai.chat_models import ChatOpenAI
import json
import logging
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, utility
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

model = SentenceTransformer(r"D:\code\rag-retrieval-main\embeding\bge-large-zh-v1.5\BAAI\bge-large-zh-v1___5",device="cuda")

connect_to_milvus()
collection_name = "csi_name"
collection = Collection(collection_name)


query = "Part 4: [2.22, 1.26, 0.89, 1.74, 1.99, 2.53, 2.54, 1.41, 2.54, 2.22, 2.54, 3.31, 2.54, 4.44, 4.44, 3.31, 2.83, 5.84, 6.09, 6.0, 5.44, 4.47, 5.81, 5.7, 6.84, 7.66, 9.36, 5.84, 7.21, 3.48]"

query_vector = [model.encode(str(query)).tolist()]
print(len(model.encode(str(query)).tolist()))
print(type(model.encode(str(query)).tolist()))
retriever = search_collection(collection, query_vector)
print(retriever[0])
print(len(retriever[0]))
retriever=str(retriever)

template = """你是一个数据分类助手，利用上下文的已分类数据，只关注上下文中'input'和'output'的内容，对用户输入的未分类数据进行分类，只需要将'output'的内容作为回答即可，不能有多余的请求和回答。
上下文:{context}
用户输入:{user_input}
answer:
"""
prompt = template.format(context=retriever, user_input=query)
#print(prompt)
#prompt = ChatPromptTemplate.from_template(template)
'''llm = ChatOpenAI(model="deepseek-v3",
    base_url=os.getenv('OPENAI_BASE_URL', 'https://xiaoai.plus/v1'),
    api_key=os.getenv('OPENAI_API_KEY'))'''
llm = ChatOpenAI(model="deepseek-chat",
    base_url=os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1'),
    api_key=os.getenv('DEEPSEEK_API_KEY'))


#answer=llm.invoke(prompt)
#print(answer.content)