import requests
import json
import time
import sys
import logging
import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection
from openai import OpenAI

load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 初始化Flask应用
app = Flask(__name__)

#使用前用docker安装milvus数据库
def connect_to_milvus():
    """连接到Milvus数据库"""
    try:
        connections.connect(
            alias="default",
            host=os.getenv('MILVUS_HOST', 'localhost'),
            port=os.getenv('MILVUS_PORT', '19530')
        )
        logger.info("成功连接至 Milvus")
        return True
    except Exception as e:
        logger.error(f"连接 Milvus 失败: {e}")
        return False

#创建搜索前需要建立数据库
def search_collection(collection, query_vector):#collection数据库名，query_vector向量查询的向量
    """在Milvus集合中搜索相似向量"""
    try:
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
    except Exception as e:
        logger.error(f"Milvus搜索失败: {e}")
        return None

#查询函数
def milvus_answer(query_list):
    """使用Milvus向量数据库检索相关信息"""
    try:
        # 加载向量嵌入模型
        model = SentenceTransformer(
            os.getenv('EMBEDDINGS_MODEL_PATH', 'BAAI/bge-large-zh-v1.5'),
            device="cuda"
        )

        # 连接Milvus
        if not connect_to_milvus():
            return []

        collection_name = "class_name_data"
        collection = Collection(collection_name)

        context = []

        # 确保query_list是列表
        if isinstance(query_list, str):
            query_list = [query_list]

        for query in query_list:
            if not query:
                continue

            # 编码查询
            query_vector = [model.encode(str(query)).tolist()]

            # 检索相似向量
            retriever = search_collection(collection, query_vector)
            if not retriever:
                context.append("")
                continue

            retriever_str = str(retriever)
            logger.info(f"Retriever结果: {retriever_str[:200]}...")  # 只记录前200个字符以避免日志过大

            # 构建提示
            template = """你是一个数据分类助手，利用上下文的已分类数据，只关注上下文中'input'和'output'的内容，对用户输入的未分类数据进行分类，只需要将'output'的内容作为回答即可，不能有多余的请求和回答。
                上下文:{context}
                用户输入:{user_input}
                answer:
                """
            prompt = template.format(context=retriever_str, user_input=query)

            client = OpenAI(
                api_key=os.getenv('DEEPSEEK_API_KEY'),
                base_url=os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1')
            )

            try:
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7
                )
                answer = response.choices[0].message.content
                logger.info(f"Milvus LLM答案: {answer}")
                context.append(answer)
            except Exception as e:
                logger.error(f"LLM调用失败: {e}")
                context.append("")

        return context
    except Exception as e:
        logger.error(f"milvus_answer函数执行失败: {e}")
        return []


def coze_answer(query):
    """通过Coze API处理查询"""
    try:
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
            "auto_save_history": not (True),
            "additional_messages": [
                {
                    "role": "user",
                    "content_type": "text",
                    "content": query
                }
            ],
            "parameters": {
                "input": "123"
            }
        }

        logger.info(f"发送到Coze的查询: {query}")
        response = requests.post(api_url, headers=headers, json=body)

        data = []
        for line in response.iter_lines():
            decoded_line = line.decode('utf-8', errors='ignore')
            if decoded_line.startswith("data:"):
                try:
                    event_data = json.loads(decoded_line[5:])
                    data.append(event_data)
                except json.JSONDecodeError:
                    continue

        if not data:
            logger.error("Coze API返回空数据")
            return []

        # 处理返回结果
        try:
            content = data[-4]['content']
            json_data = json.loads(content)

            if json_data.get("nottimes"):
                identity_list = json_data["nottimes"]
            else:
                identity_list = json_data.get("realtime", [])

            # 确保返回值是列表
            if isinstance(identity_list, str):
                identity_list = [identity_list]

            logger.info(f"Coze处理后的查询列表: {identity_list}")
            return identity_list

        except Exception as e:
            logger.error(f"解析Coze响应失败: {e}")
            # 如果无法解析，直接返回原始查询作为备选
            return [query]

    except Exception as e:
        logger.error(f"coze_answer函数执行失败: {e}")
        # 返回原始查询作为备选
        return [query]


def final_answer(query, context):
    """生成最终回答"""
    try:
        # 如果context是空列表，返回默认回答
        if not context or all(c == "" for c in context):
            return "抱歉，我没有找到相关信息。请尝试换一种方式提问。"

        context_str = "; ".join([c for c in context if c])

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

        prompt = template.format(query=query, context=context_str)
        logger.info(f"最终提示: {prompt}")

        # 使用 OpenAI 库生成答案
        client = OpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('OPENAI_BASE_URL', 'https://xiaoai.plus/v1')
        )

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        full_content = response.choices[0].message.content
        logger.info(f"LLM完整回答: {full_content}")

        # 提取回答部分
        try:
            data = full_content.split("回答：")[1].strip()
        except IndexError:
            # 如果无法拆分，返回完整内容
            data = full_content

        return data

    except Exception as e:
        logger.error(f"final_answer函数执行失败: {e}")
        return "抱歉，处理您的查询时出现了问题。请稍后再试。"


def chat(query):
    """处理完整的聊天流程，返回回答和调试信息"""
    logger.info(f"收到新查询: {query}")

    try:
        # 步骤1: 通过Coze处理查询
        query_list = coze_answer(query)

        # 步骤2: 通过Milvus获取上下文
        context = milvus_answer(query_list)

        # 步骤3: 生成最终回答
        answer = final_answer(query, context)

        # 返回结果和调试信息
        return {
            "answer": answer,
            "debug": {
                "coze_queries": query_list,
                "milvus_context": context
            }
        }

    except Exception as e:
        logger.error(f"chat函数执行失败: {e}")
        return {
            "answer": "抱歉，处理您的查询时出现了问题。请稍后再试。",
            "debug": {
                "error": str(e)
            }
        }


@app.route('/api/chat', methods=['POST'])
def handle_chat():
    """处理聊天API请求"""
    try:
        data = request.json
        query = data.get('query', '')

        if not query:
            return jsonify({"error": "查询不能为空"}), 400

        result = chat(query)
        return jsonify(result)

    except Exception as e:
        logger.error(f"API请求处理失败: {e}")
        return jsonify({
            "answer": "服务器处理请求时出错",
            "error": str(e)
        }), 500


# 创建静态目录
@app.route('/')
def index():
    return send_file('static/index.html')


# 处理其他静态文件请求
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)


# 简单的辅助函数来发送文件
def send_file(filename):
    try:
        root_dir = os.path.dirname(os.path.abspath(__file__))
        return send_from_directory(os.path.join(root_dir, 'static'), os.path.basename(filename))
    except Exception as e:
        logger.error(f"发送文件失败: {e}")
        return "文件未找到", 404


# CLI测试函数
def test_cli():
    """命令行测试功能"""
    query = input("请输入查询: ")
    result = chat(query)
    print("\n回答:", result["answer"])
    print("\n调试信息:")
    print("Coze查询:", result["debug"]["coze_queries"])
    print("Milvus上下文:", result["debug"]["milvus_context"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='智能问答系统')
    parser.add_argument('--mode', type=str, default='server', choices=['server', 'cli'],
                        help='运行模式: server (Web服务) 或 cli (命令行界面)')
    parser.add_argument('--port', type=int, default=5000, help='Web服务器端口')
    parser.add_argument('--debug', action='store_true', help='是否启用Flask调试模式')

    args = parser.parse_args()

    if args.mode == 'server':
        logger.info(f"启动Web服务，端口: {args.port}")

        # 确保静态目录存在
        static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
        if not os.path.exists(static_dir):
            os.makedirs(static_dir)
            logger.info(f"创建静态目录: {static_dir}")

        # 将HTML文件写入静态目录
        index_html_path = os.path.join(static_dir, 'index.html')
        with open(index_html_path, 'w', encoding='utf-8') as f:
            f.write("""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>智能问答系统</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/axios/1.3.4/axios.min.js"></script>
    <style>
        body {
            font-family: 'PingFang SC', 'Microsoft YaHei', sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
            color: #333;
        }
        .container {
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            height: 100vh;
        }
        .header {
            background-color: #1890ff;
            color: white;
            padding: 15px 20px;
            border-radius: 8px 8px 0 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 {
            margin: 0;
            font-size: 1.5rem;
        }
        .chat-area {
            flex: 1;
            background-color: white;
            border-left: 1px solid #e8e8e8;
            border-right: 1px solid #e8e8e8;
            overflow-y: auto;
            padding: 20px;
        }
        .message {
            margin-bottom: 15px;
            display: flex;
        }
        .user-message {
            justify-content: flex-end;
        }
        .bot-message {
            justify-content: flex-start;
        }
        .message-content {
            max-width: 70%;
            padding: 10px 15px;
            border-radius: 18px;
            word-wrap: break-word;
        }
        .user-message .message-content {
            background-color: #1890ff;
            color: white;
            border-bottom-right-radius: 4px;
        }
        .bot-message .message-content {
            background-color: #f0f0f0;
            color: #333;
            border-bottom-left-radius: 4px;
        }
        .input-area {
            display: flex;
            padding: 15px;
            background-color: white;
            border-top: 1px solid #e8e8e8;
            border-radius: 0 0 8px 8px;
        }
        .input-area input {
            flex: 1;
            padding: 10px 15px;
            border: 1px solid #d9d9d9;
            border-radius: 4px;
            font-size: 1rem;
            margin-right: 10px;
        }
        .input-area button {
            background-color: #1890ff;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 10px 20px;
            cursor: pointer;
            font-size: 1rem;
        }
        .input-area button:hover {
            background-color: #40a9ff;
        }
        .input-area button:disabled {
            background-color: #bfbfbf;
            cursor: not-allowed;
        }
        .typing-indicator {
            display: flex;
            padding: 10px 15px;
        }
        .typing-indicator span {
            height: 8px;
            width: 8px;
            background-color: #bbb;
            border-radius: 50%;
            display: inline-block;
            margin: 0 2px;
            animation: typing 1s infinite;
        }
        .typing-indicator span:nth-child(2) {
            animation-delay: 0.2s;
        }
        .typing-indicator span:nth-child(3) {
            animation-delay: 0.4s;
        }
        @keyframes typing {
            0% { transform: translateY(0); }
            50% { transform: translateY(-5px); }
            100% { transform: translateY(0); }
        }
        .system-message {
            text-align: center;
            color: #999;
            margin: 10px 0;
            font-size: 0.9rem;
        }
        .debug-panel {
            background-color: #f9f9f9;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 10px;
            margin-top: 10px;
            display: none;
        }
        .debug-toggle {
            background: none;
            border: none;
            color: #1890ff;
            cursor: pointer;
            font-size: 0.8rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>智能信息查询助手</h1>
            <button class="debug-toggle" onclick="toggleDebug()">调试面板</button>
        </div>
        <div class="chat-area" id="chatArea">
            <div class="message bot-message">
                <div class="message-content">
                    您好！我是您的智能助手，可以帮您查询相关信息。请问有什么可以帮您？
                </div>
            </div>
        </div>
        <div class="debug-panel" id="debugPanel">
            <h3>调试信息</h3>
            <div id="debugInfo"></div>
        </div>
        <div class="input-area">
            <input type="text" id="userInput" placeholder="请输入您的问题..." onkeypress="handleKeyPress(event)">
            <button id="sendButton" onclick="sendMessage()">发送</button>
        </div>
    </div>

    <script>
        let isProcessing = false;

        function toggleDebug() {
            const debugPanel = document.getElementById('debugPanel');
            if (debugPanel.style.display === 'none' || debugPanel.style.display === '') {
                debugPanel.style.display = 'block';
            } else {
                debugPanel.style.display = 'none';
            }
        }

        function appendMessage(content, isUser) {
            const chatArea = document.getElementById('chatArea');
            const messageDiv = document.createElement('div');
            messageDiv.className = isUser ? 'message user-message' : 'message bot-message';

            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            contentDiv.textContent = content;

            messageDiv.appendChild(contentDiv);
            chatArea.appendChild(messageDiv);
            chatArea.scrollTop = chatArea.scrollHeight;
        }

        function appendDebugInfo(info) {
            const debugInfo = document.getElementById('debugInfo');
            const p = document.createElement('p');
            p.textContent = info;
            debugInfo.appendChild(p);
        }

        function showTypingIndicator() {
            const chatArea = document.getElementById('chatArea');
            const typingDiv = document.createElement('div');
            typingDiv.className = 'message bot-message';
            typingDiv.id = 'typingIndicator';

            const typingContent = document.createElement('div');
            typingContent.className = 'typing-indicator';
            for (let i = 0; i < 3; i++) {
                const dot = document.createElement('span');
                typingContent.appendChild(dot);
            }

            typingDiv.appendChild(typingContent);
            chatArea.appendChild(typingDiv);
            chatArea.scrollTop = chatArea.scrollHeight;
        }

        function removeTypingIndicator() {
            const typingIndicator = document.getElementById('typingIndicator');
            if (typingIndicator) {
                typingIndicator.remove();
            }
        }

        function sendMessage() {
            if (isProcessing) return;

            const userInput = document.getElementById('userInput');
            const query = userInput.value.trim();

            if (query === '') return;

            appendMessage(query, true);
            userInput.value = '';

            isProcessing = true;
            document.getElementById('sendButton').disabled = true;
            showTypingIndicator();

            // API调用
            axios.post('/api/chat', { query: query })
                .then(response => {
                    removeTypingIndicator();
                    appendMessage(response.data.answer, false);

                    // 显示调试信息
                    if (response.data.debug) {
                        appendDebugInfo(`Coze 查询: ${JSON.stringify(response.data.debug.coze_queries)}`);
                        appendDebugInfo(`Milvus 上下文: ${JSON.stringify(response.data.debug.milvus_context)}`);
                    }
                })
                .catch(error => {
                    removeTypingIndicator();
                    appendMessage('抱歉，处理您的请求时出现了问题，请稍后再试。', false);
                    console.error('Error:', error);
                })
                .finally(() => {
                    isProcessing = false;
                    document.getElementById('sendButton').disabled = false;
                });
        }

        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }
    </script>
</body>
</html>""")
        logger.info(f"写入HTML文件至: {index_html_path}")

        app.run(host='0.0.0.0', port=args.port, debug=args.debug)
    else:
        logger.info("启动命令行测试模式")
        test_cli()