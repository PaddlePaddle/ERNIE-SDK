import argparse
import os
import glob
import json

import erniebot
import gradio as gr
from pipelines.document_stores import FAISSDocumentStore
from pipelines.nodes import (
    CharacterTextSplitter,
    SpacyTextSplitter,
    EmbeddingRetriever,
    ErnieBot,
    ErnieRanker,
    PDFToTextConverter,
    PromptTemplate,
)
from pipelines.pipelines import Pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--index_name", default='construct_demo_index', type=str, help="The ann index name of ANN.")
parser.add_argument("--file_paths", default='./construction_regulations', type=str, help="The PDF file path.")
parser.add_argument("--retriever_top_k", default=5, type=int, help="Number of recall items for search")
parser.add_argument("--chunk_size", default=384, type=int, help="The length of data for indexing by retriever")
parser.add_argument('--host', type=str, default="localhost", help='host ip of ANN search engine')
parser.add_argument('--port', type=int, default=8081, help='host ip of ANN search engine')
parser.add_argument("--api_key", default=None, type=str, help="The API Key.")
parser.add_argument("--secret_key", default=None, type=str, help="The secret key.")
args = parser.parse_args()

erniebot.api_type = "qianfan"
erniebot.ak = args.api_key
erniebot.sk = args.secret_key

# 利用Paddle-Pipelines构建本地语义检索服务
faiss_document_store = "faiss_document_store.db"
# 如本地语义检索索引已经存在，则直接读取已经构建好的索引
if os.path.exists(args.index_name) and os.path.exists(faiss_document_store):
    # connect to existed FAISS Index
    document_store = FAISSDocumentStore.load(args.index_name)
    retriever = EmbeddingRetriever(
        document_store=document_store,
        retriever_batch_size=16,  # 16 is the max batch size allowed by ErnieBot Embedding
        api_key=args.api_key,
        secret_key=args.secret_key,
    )
# 如本地语义检索索引已经存在，则直接读取已经构建好的索引
else:
    if os.path.exists(args.index_name):
        os.remove(args.index_name)
    if os.path.exists(faiss_document_store):
        os.remove(faiss_document_store)
    document_store = FAISSDocumentStore(
        embedding_dim=384,  # hardcode the embedding dim to 384 for ErnieBot Embedding
        duplicate_documents="skip",
        return_embedding=True,
        faiss_index_factory_str="Flat",
    )
    retriever = EmbeddingRetriever(
        document_store=document_store,
        retriever_batch_size=16,  # 16 is the max batch size allowed by ErnieBot Embedding
        api_key=args.api_key,
        secret_key=args.secret_key,
    )
    # 将Word文档转换为文字
    pdf_converter = PDFToTextConverter()
    # 将文字分片
    text_splitter = SpacyTextSplitter(separator="\n", chunk_size=384, chunk_overlap=128, filters=["\n"])
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_node(component=pdf_converter, name="pdf_converter", inputs=["File"])
    indexing_pipeline.add_node(component=text_splitter, name="Splitter", inputs=["pdf_converter"])
    indexing_pipeline.add_node(component=retriever, name="Retriever", inputs=["Splitter"])
    indexing_pipeline.add_node(component=document_store, name="DocumentStore", inputs=["Retriever"])
    files_paths = glob.glob(args.file_paths + "/*.pdf")
    indexing_pipeline.run(file_paths=files_paths)
    document_store.save(args.index_name)

# 构建用于查询的Pipelines
ernie_bot = ErnieBot(api_key=args.api_key, secret_key=args.secret_key)
query_pipeline = Pipeline()
query_pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])

# 定义函数api, 共1个api, 以及2个使用例子
functions = [
    {
        "name": "search_knowledge_base",
        "description": "在住房和城乡建设部规章中寻找和query最相关的片段",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "规章查询语句"}},
            "required": ["query"],
        },
        "responses": {
            "type": "object",
            "description": "检索结果，内容为住房和城乡建设部规章中和query相关的文字片段",
            "properties": {
                "documents": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "document": {"type": "string", "description": "和query相关的文字片段"},
                        },
                    },
                }
            },
            "required": ["documents"],
        },
        "examples": [
            {"role": "user", "content": "企业申请建筑业企业资质需要哪些材料？"},
            {
                "role": "assistant",
                "content": None,
                "function_call": {
                    "name": "search_knowledge_base",
                    "thoughts": "这是和城市建设法规标准相关的问题，我需要查询住房和城乡建设部规章，并且设置query为'企业申请建筑业企业资质需要的材料'",
                    "arguments": '{ "query": "企业申请建筑业企业资质需要哪些材料？"}',
                },
            },
            {"role": "user", "content": "历史文化街区的城市设计有什么要求？"},
            {
                "role": "assistant",
                "content": None,
                "function_call": {
                    "name": "search_knowledge_base",
                    "thoughts": "这是和城市建设法规标准相关的问题，我需要查询住房和城乡建设部规章，并且设置query为'历史文化街区的设计要求'",
                    "arguments": '{ "query": "历史文化街区的设计要求"}',
                },
            },
        ],
    }
]


def search_knowledge_base(query):
    prediction = query_pipeline.run(
        query=query,
        params={
            "Retriever": {
                "top_k": args.retriever_top_k,
            },
        },
    )
    documents = [{ "document": doc.content } for doc in prediction["documents"]]
    return {"documents": documents}

def history_transform(history=[]):
    messages = []
    if len(history) < 2:
        return messages

    for turn_idx in range(1, len(history)):
        messages.extend(
            [{"role": "user", "content": history[turn_idx][0]}, {"role": "assistant", "content": history[turn_idx][1]}]
        )
    return messages


def add_message_chatbot(messages, history):
    history.append([messages, None])
    return None, history


def prediction(history):
    logs = []
    query = history.pop()[0]
    if query == "":
        return history, "注意：问题不能为空"
    
    # 消除潜在的错误格式问题
    for turn_idx in range(len(history)):
        if history[turn_idx][0] is not None:
            history[turn_idx][0] = history[turn_idx][0].replace("<br>", "")
        if history[turn_idx][1] is not None:
            history[turn_idx][1] = history[turn_idx][1].replace("<br>", "")

    # 将对话历史从gradio格式转化为 eb sdk的格式
    messages = history_transform(history)
    # 插入将当前轮次的用户query插入上下文当中
    messages.append({"role": "user", "content": query})
    # 调用eb的chat completion, 提供functions入参
    response = erniebot.ChatCompletion.create(
        model="ernie-bot-3.5",
        messages=messages,
        functions=functions,
    )
    # 如果function call未触发，模型直接回答，则直接返回模型结果
    if "function_call" not in response:
        logs.append({"function_call结果": "未触发"})
        result = response["result"]
    # 如果function call触发
    else:
        function_call = response.function_call
        logs.append({"function_call结果": function_call})
        # 解析模型返回的function call入参
        func_args = json.loads(function_call["arguments"])
        # 调用function
        res = search_knowledge_base(**func_args)
        logs.append({"检索结果": res})
        # 使用 eb的chat completion 对于function返回的结果进行润色, 用结果回复用户
        messages.append({"role": "assistant", "content": None, "function_call": function_call})
        messages.append(
            {"role": "function", "name": function_call["name"], "content": json.dumps(res, ensure_ascii=False)}
        )
        response = erniebot.ChatCompletion.create(model="ernie-bot-3.5", messages=messages)
        result = response["result"]
    history.append([query, result])
    return history, logs


def launch_ui():
    with gr.Blocks(title="ERNIE Bot 城市建设法规标准小助手", theme=gr.themes.Base()) as demo:
        gr.HTML("""<h1 align="center">ERNIE Bot 城市建设法规标准小助手</h1>""")
        with gr.Column():
            chatbot = gr.Chatbot(value=[[None, "您好, 我是 ERNIE Bot 城市建设法规标准小助手。除了普通的大模型能力以外，还特别了解住房和城乡建设部规章哦"]], scale=35, height=500)
            message = gr.Textbox(placeholder="哪些建筑企业资质需要国务院住房城乡建设主管部门许可？", lines=1, max_lines=20)
            with gr.Row():
                submit = gr.Button("🚀 提交", variant="primary", scale=1)
                clear = gr.Button("清除", variant="primary", scale=1)
            log = gr.JSON()
        message.submit(add_message_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]).then(
            prediction, inputs=[chatbot], outputs=[chatbot, log]
        )
        submit.click(add_message_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]).then(
            prediction, inputs=[chatbot], outputs=[chatbot, log]
        )
        clear.click(lambda _: ([[None, "您好, 我是 ERNIE Bot 城市建设法规标准小助手。除了普通的大模型能力以外，还特别了解住房和城乡建设部规章哦"]]), inputs=[clear], outputs=[chatbot])
    demo.launch(server_name=args.host, server_port=args.port, debug=True)

if __name__ == "__main__":
    launch_ui()
