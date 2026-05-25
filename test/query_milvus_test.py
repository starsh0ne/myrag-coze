import os
import json
import logging
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()
from pymilvus import connections, CollectionSchema, FieldSchema, DataType, Collection, utility
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
import ollama
import re

def load_config(config_path="config.json"):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logging.info("配置文件加载成功")
        return config
    except Exception as e:
        logging.error(f"加载配置文件失败: {e}")
        exit(1)

def connect_to_milvus(config):
    try:
        connections.connect(
            alias="default",
            host=config["milvus"]["host"],
            port=config["milvus"]["port"]
        )
        logging.info("成功连接至 Milvus")
    except Exception as e:
        logging.error(f"连接 Milvus 失败: {e}")
        exit(1)

def load_collection(collection):
    try:
        collection.load()
        logging.info("Collection loaded into memory.")
    except Exception as e:
        logging.error(f"Error loading collection: {e}")
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

def prompt_function(context, question):
    prompt_template = """你是一个数据分类助手，利用上下文的已分类数据，只关注上下文中'input'和'output'的内容，对未分类数据进行分类，只需要将'output'的内容作为回答即可，不能有多余的请求和回答。
    上下文:{context}
    未分类数据:{question}
    """
    return prompt_template.format(context=context, question=question)


def query_file(llm,json_file):
    try:
        with open(json_file, 'r') as file:
            articles = json.load(file)
        logging.info("JSON file loaded successfully.")
    except FileNotFoundError:
        logging.error(f"The file {json_file} was not found.")
        articles = []
    except json.JSONDecodeError:
        logging.error(f"The file {json_file} does not contain valid JSON data.")
        articles = []
    i = 0
    for article in articles:
        query = article["input"]
        # query = get_querylist(query)
        query_vector = [model.encode(str(query)).tolist()]

        search_results = search_collection(collection, query_vector)
        # prompt_text = prompt_function(search_results, query)
        # result = llm3.invoke(prompt_text).content

    print(i / len(articles))


if __name__ == "__main__":
    config=load_config(config_path="config.json")
    model = SentenceTransformer(os.getenv('EMBEDDINGS_MODEL_PATH', 'BAAI/bge-large-zh-v1.5'), device="cuda")

    connect_to_milvus(config)
    collection_name = config.get("milvus").get("collection_name")
    collection = Collection(collection_name)

    llm = ChatOllama(
        model="deepseek7b:latest",  # 使用本地部署的 Llama2 模型
        base_url="http://localhost:11434"  # Ollama 的本地服务地址
    )
    # query_file(llm,json_file)
    query = "Part 4: [2.22, 1.26, 0.89, 1.74, 1.99, 2.53, 2.54, 1.41, 2.54, 2.22, 2.54, 3.31, 2.54, 4.44, 4.44, 3.31, 2.83, 5.84, 6.09, 6.0, 5.44, 4.47, 5.81, 5.7, 6.84, 7.66, 9.36, 5.84, 7.21, 3.48]"

    query_vector = [model.encode(str(query)).tolist()]

    search_results = search_collection(collection, query_vector)
    print(search_results)
    data=str(search_results[0][0])
    result=re.search(r"output:(\w+)", data)

    print(result.group(1))
    #res = ollama.chat(model="deepseek7b:latest", stream=False, messages=[{"role": "user", "content": query}],options={"temperature": 0})
    #print(res)