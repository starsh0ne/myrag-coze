import json
import logging
import os
import csv
from datetime import datetime
from dotenv import load_dotenv
import torch
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline

load_dotenv()
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_optimized.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ResultRecorder:
    """优化后的结果记录器，批量写入CSV文件"""

    def __init__(self, output_file="dsv3_fe_ma.csv"):
        self.output_file = output_file
        self.results = []  # 存储所有结果
        self._initialize_csv()

    def _initialize_csv(self):
        """初始化CSV文件"""
        with open(self.output_file, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Timestamp', 'Filename', 'Sample Index',
                'LLM Answer', 'Real Answer', 'Is Correct',
                'Current Accuracy', 'File Accuracy', 'Total Samples'
            ])

    def add_result(self, data):
        """添加单条结果到内存"""
        self.results.append(data)

    def write_batch(self):
        """批量写入结果到CSV文件"""
        try:
            with open(self.output_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(self.results)
            self.results = []  # 清空已写入的结果
        except Exception as e:
            logger.error(f"批量记录结果失败: {e}")


def get_device():
    """安全获取可用设备"""
    try:
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            logger.info(f"发现 {device_count} 个CUDA设备")
            return torch.device("cuda:0")  # 显式使用GPU 1
        return torch.device("cpu")
    except Exception as e:
        logger.error(f"获取设备时出错: {e}, 将使用CPU")
        return torch.device("cpu")


def connect_to_milvus():
    """连接到Milvus数据库"""
    try:
        connections.connect(
            alias="default",
            host='localhost',
            port=19530
        )
        logger.info("成功连接至Milvus")
        return True
    except Exception as e:
        logger.error(f"连接Milvus失败: {e}")
        return False


def search_collection(collection, query_vector, top_k=5):
    """在Milvus集合中搜索相似向量"""
    try:
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }
        return collection.search(
            data=[query_vector],
            anns_field="vectors",
            param=search_params,
            limit=top_k,
            output_fields=["metadata"]
        )
    except Exception as e:
        logger.error(f"Milvus搜索失败: {e}")
        return None


def extract_answer(llm_output: str) -> str:
    """从LLM输出中提取分类结果"""
    answer_prefix = "回答:"
    last_idx = llm_output.rfind(answer_prefix)

    if last_idx == -1:
        return ""  # 未找到答案

    answer = llm_output[last_idx + len(answer_prefix):].strip()
    return answer.split("\n")[0].split("|")[0].strip()


def process_file(data, embedding_model, collection, template, llm, filename, recorder):
    """处理单个文件，优化为批量写入结果"""
    correct = 0
    total = 0
    file_results = []  # 存储当前文件的所有结果

    for idx, query_data in enumerate(data):
        if not query_data.get('input'):
            continue

        try:
            # 编码查询
            query_vector = embedding_model.encode(query_data['input']).tolist()

            # 检索相似向量
            search_results = search_collection(collection, query_vector)
            if not search_results:
                raise ValueError("搜索返回空结果")

            # 构建提示并获取LLM回答
            '''prompt = {
                "context": str(search_results[0]),
                "user_input": query_data['input']
            }

            llm_answer = llm_chain.invoke(prompt)
            llm_answer = extract_answer(llm_answer)'''
            prompt = template.format(
                context=str(search_results[0]),
                user_input=query_data['input']
            )
            llm_answer = llm.invoke(prompt).content.strip()
            real_answer = query_data['output']

            # 评估结果
            is_correct = llm_answer.strip() == real_answer.strip()
            correct += int(is_correct)
            total += 1
            current_accuracy = (correct / total * 100) if total > 0 else 0

            # 记录日志
            logger.info(
                f"{filename} - 样本 {idx + 1}/{len(data)}: "
                f"LLM: {llm_answer} | 真实: {real_answer} | "
                f"正确: {'是' if is_correct else '否'} | "
                f"当前准确率: {current_accuracy:.2f}%"
            )

            # 将结果添加到内存中
            file_results.append([
                #datetime.now().isoformat(),
                0,
                filename,
                idx,
                llm_answer,
                real_answer,
                int(is_correct),
                current_accuracy,
                0,  # 文件准确率将在最后更新
                total
            ])

        except Exception as e:
            logger.error(f"处理样本 {idx} 失败: {e}")
            file_results.append([
                #datetime.now().isoformat(),
                0,
                filename,
                idx,
                "",
                query_data.get('output', ''),
                0,
                0,
                0,
                total
            ])
            continue

    # 计算文件整体准确率
    file_accuracy = (correct / total * 100) if total > 0 else 0

    # 更新文件结果中的文件准确率
    for result in file_results:
        result[7] = file_accuracy  # 更新文件准确率字段

    # 添加文件总结行
    file_results.append([
        #datetime.now().isoformat(),
        0,
        filename,
        'SUMMARY',
        '',
        '',
        '',
        0,
        file_accuracy,
        total
    ])

    # 批量添加结果到记录器
    recorder.results.extend(file_results)

    # 写入当前文件的所有结果
    recorder.write_batch()

    logger.info(
        f"\n文件 {filename} 处理完成\n"
        f"总样本数: {total}\n"
        f"正确数: {correct}\n"
        f"准确率: {file_accuracy:.2f}%\n"
        "----------------------------------------"
    )

    return file_results, file_accuracy


def milvus_answer(directory):
    """主处理函数"""
    try:
        # 初始化
        device = get_device()
        recorder = ResultRecorder()

        # 加载向量嵌入模型
        embedding_model = SentenceTransformer(
            r"D:\models\bge-large-zh-v1.5\BAAI\bge-large-zh-v1___5",
            device=device
        )

        llm = ChatOpenAI(
            model="deepseek-chat",
            base_url=os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1'),
            api_key=os.getenv('DEEPSEEK_API_KEY'),
            timeout=30
        )
        template = """你是一个数据分类助手，利用上下文的已分类数据，只关注上下文中'input'和'output'的内容，对用户输入的未分类数据进行分类，只需要将'output'的内容作为回答即可，不能有多余的请求和回答。
        上下文:{context}
        用户输入:{user_input}
        回答:"""

        # 连接Milvus
        if not connect_to_milvus():
            raise ConnectionError("无法连接Milvus数据库")

        collection = Collection("class_name_data")
        collection.load()

        # 处理目录中的文件
        if not os.path.exists(directory):
            raise FileNotFoundError(f"目录不存在: {directory}")

        for filename in sorted(os.listdir(directory)):
            if filename.endswith('.json'):
                file_path = os.path.join(directory, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        results, accuracy = process_file(
                            data, embedding_model, collection,
                            template, llm, filename, recorder
                        )
                except Exception as e:
                    logger.error(f"处理文件 {filename} 失败: {e}")

    except Exception as e:
        logger.error(f"主处理函数执行失败: {e}", exc_info=True)
    finally:
        # 清理资源
        try:
            connections.disconnect("default")
            logger.info("已断开Milvus连接")
        except:
            pass


def main():
    directory_path = r'D:\code\rag-retrieval-main\chat_ui\new_data\dealed\test' #测试集路径，json文件
    milvus_answer(directory_path)


if __name__ == '__main__':
    main()