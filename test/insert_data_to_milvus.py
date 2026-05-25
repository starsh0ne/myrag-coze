import os
import json
import logging
from dotenv import load_dotenv
import torch
from sentence_transformers import SentenceTransformer
from pymilvus import connections, CollectionSchema, FieldSchema, DataType, Collection, utility
from langchain_core.documents import Document

load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO)

config_file = os.getenv('CONFIG_FILE_PATH', 'config.json')

# 读取JSON文件
with open(config_file, 'r', encoding='utf-8') as file:
    config = json.load(file)

# 获取基础路径
base_path = config['path']
# 获取训练文件名
train_files = config['train_file']

# 构建完整的文件路径


def open_load_files(json_file_path):
    try:
        with open(json_file_path, 'r') as file:
            articles = json.load(file)
        logging.info("JSON file loaded successfully.")
        return articles
    except FileNotFoundError:
        logging.error(f"The file {json_file_path} was not found.")
        articles = []
    except json.JSONDecodeError:
        logging.error(f"The file {json_file_path} does not contain valid JSON data.")
        articles = []



def split_text(text):
    parts = text.split("|")
    data_dict = {}
    for part in parts:
        # 找到冒号和方括号的位置
        colon_index = part.find(":")
        start_bracket_index = part.find("[")
        end_bracket_index = part.find("]")
        if colon_index != -1 and start_bracket_index != -1 and end_bracket_index != -1:
            # 提取部分名称
            part_name = part[:colon_index].strip()
            # 提取方括号内的列表数据
            list_str = part[start_bracket_index + 1:end_bracket_index]
            # 将字符串转换为列表
            data_list = [float(num) for num in list_str.split(",") if num.strip()]
            #print(data_list)
            data_dict[part_name] = data_list
    return data_dict


def combined_text(data):
    combined_list = []
    for sub_list in data.values():
        combined_list.extend(sub_list)
    return combined_list


def add_vectors_to_articles(articles, model):

    updated_articles = []
    if not articles:
        return updated_articles
    for article in articles:
        if 'input' not in article:
            logging.warning("Article is missing 'input' field. Skipping...")
            continue
        try:
            #splited_text = split_text(article['input'])
            new_article = {
                'vectors': model.encode(str(article['input'])).tolist(),
                'metadata': "input:"+str(article['input'])+" output:"+article.get("output", "")
            }
            updated_articles.append(new_article)
        except Exception as e:
            logging.error(f"Error processing article: {e}. Skipping...")
    return updated_articles


def connect_to_milvus():
    try:
        connections.connect(alias="default", host="localhost", port="19530")
        logging.info("Connected to Milvus successfully.")
    except Exception as e:
        logging.error(f"Failed to connect to Milvus: {e}")
        exit(1)


def create_collection_with_metadata(collection_name):
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vectors", dtype=DataType.FLOAT_VECTOR, dim=1024),
        FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=1024)
    ]
    schema = CollectionSchema(fields, description="CSI Embeddings Collection")
    try:
        if not utility.has_collection(collection_name):
            collection = Collection(name=collection_name, schema=schema)
            logging.info(f"Collection '{collection_name}' created.")
        else:
            collection = Collection(name=collection_name)
            logging.info(f"Collection '{collection_name}' already exists.")
    except Exception as e:
        logging.error(f"Error creating collection: {e}")
        exit(1)
    return collection


def insert_data_with_metadata(collection, data, batch_size=1000):
    if not data:
        return
    try:
        collection.insert(data)
        logging.info(f"Batch of {len(data)} data inserted into collection successfully.")
    except Exception as e:
        logging.error(f"Error inserting batch of data into collection: {e}")


def create_index(collection):
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 100}
    }
    try:
        collection.create_index(field_name="vectors", index_params=index_params)

        logging.info(f"Index created on vector fields.")
    except Exception as e:
        logging.error(f"Error creating index: {e}")
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

def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    model = SentenceTransformer(os.getenv('EMBEDDINGS_MODEL_PATH', 'BAAI/bge-large-zh-v1.5'), device=device)
    for key, filename in train_files.items():
        full_path = f"{base_path}\\{filename}"
        print(f"File {key}: {full_path}")
        articles = open_load_files(full_path)
        updated_articles = add_vectors_to_articles(articles, model)
        connect_to_milvus()
        collection = Collection(name="csi_name")  # collection 存在时
        # collection = create_collection_with_metadata("csi_embeddings")#collection 不存在时
        insert_data_with_metadata(collection, updated_articles)
        create_index(collection)
        load_collection(collection)




if __name__ == "__main__":
    #run()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    model = SentenceTransformer(os.getenv('EMBEDDINGS_MODEL_PATH', 'BAAI/bge-large-zh-v1.5'), device=device)
    #json_file_path=base_path+train_files["filename11"]
    json_file_path ='../data/people4.json'
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            articles = json.load(file)
        logging.info("JSON file loaded successfully.")
    except FileNotFoundError:
        logging.error(f"The file {json_file_path} was not found.")
        articles = []
    print(articles)
    updated_articles = add_vectors_to_articles(articles, model)
    if not updated_articles:
        logging.info("No articles to process.")
    else:
        connect_to_milvus()
        if utility.has_collection("csi_name"):
            # 删除集合
            #utility.drop_collection("csi_embeddings")
            print("集合 'csi_name' 已成功删除。")
        else:
            print("集合 'csi_name' 不存在，无需删除。")
        collection = Collection(name="csi_name")#collection 存在时
        #collection = create_collection_with_metadata("csi_name")#collection 不存在时
        insert_data_with_metadata(collection, updated_articles)
        create_index(collection)
        load_collection(collection)
        query='[25.16, 33.09, 35.37, 34.76, 33.67, 30.69, 30.9, 31.52, 32.98, 37.08, 38.05, 38.97, 38.46, 37.09, 36.39, 38.67, 41.17, 44.37, 46.13, 47.14, 45.63, 43.43, 43.07, 43.28, 42.3, 47.2, 50.67, 47.93, 43.49, 37.99]'
        query_vector = [model.encode(str(query)).tolist()] # 应该是 vector

