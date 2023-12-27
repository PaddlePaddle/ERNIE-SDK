import argparse
import asyncio

import erniebot
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import SpacyTextSplitter
from tqdm import tqdm

from erniebot_agent.agents import (
    FunctionAgentWithRetrieval,
    FunctionAgentWithRetrievalScoreTool,
    FunctionAgentWithRetrievalTool,
)
from erniebot_agent.chat_models import ERNIEBot
from erniebot_agent.memory import WholeMemory
from erniebot_agent.retrieval import BaizhongSearch
from erniebot_agent.retrieval.document import Document
from erniebot_agent.tools import RemoteToolkit
from erniebot_agent.tools.baizhong_tool import BaizhongSearchTool

parser = argparse.ArgumentParser()
parser.add_argument("--base_url", type=str, help="The Aurora serving path.")
parser.add_argument("--data_path", default="construction_regulations", type=str, help="The data path.")
parser.add_argument(
    "--access_token", default="ai_studio_access_token", type=str, help="The aistudio access token."
)
parser.add_argument("--api_type", default="qianfan", type=str, help="The aistudio access token.")
parser.add_argument("--api_key", default="", type=str, help="The API Key.")
parser.add_argument("--secret_key", default="", type=str, help="The secret key.")
parser.add_argument("--indexing", action="store_true", help="The indexing step.")
parser.add_argument("--project_id", default=-1, type=int, help="The API Key.")
parser.add_argument(
    "--retrieval_type",
    choices=["rag", "rag_tool", "rag_threshold"],
    default="rag",
    help="Retrieval type, default to rag.",
)
parser.add_argument("--knowledge_base_id", default="", type=str, help="The knowledge base id.")
parser.add_argument(
    "--knowledge_base_name", default="knowledge_base_name", type=str, help="The knowledge base name."
)
args = parser.parse_args()


def offline_ann(data_path, baizhong_db):
    loader = PyPDFDirectoryLoader(data_path)
    documents = loader.load()
    text_splitter = SpacyTextSplitter(pipeline="zh_core_web_sm", chunk_size=1500, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    list_data = []
    for item in tqdm(docs):
        doc_title = item.metadata["source"].split("/")[-1]
        doc_content = item.page_content
        example = {"title": doc_title, "content_se": doc_content}
        example = Document.from_dict(example)
        list_data.append(example)
    res = baizhong_db.add_documents(documents=list_data)
    return res


if __name__ == "__main__":
    erniebot.api_type = args.api_type
    erniebot.access_token = args.access_token
    baizhong_db = BaizhongSearch(
        knowledge_base_name=args.knowledge_base_name,
        access_token=args.access_token,
        knowledge_base_id=args.knowledge_base_id if args.knowledge_base_id != "" else None,
    )
    print(baizhong_db.knowledge_base_id)
    if args.indexing:
        res = offline_ann(args.data_path, baizhong_db)
        print(res)

    llm = ERNIEBot(
        model="ernie-bot", api_type="aistudio", enable_multi_step_tool_call=True, enable_citation=True
    )

    retrieval_tool = BaizhongSearchTool(
        description="Use Baizhong Search to retrieve documents.", db=baizhong_db, threshold=0.1
    )

    # agent = FunctionAgentWithRetrievalTool(
    #     llm=llm, knowledge_base=baizhong_db, top_k=3, tools=[NotesTool(), retrieval_tool], memory=memory
    # )

    # queries = [
    #     "请把飞桨这两个字添加到笔记本中",
    #     "OpenAI管理层变更会带来哪些影响？并请把搜索的内容添加到笔记本中",
    #     "OpenAI管理层变更会带来哪些影响?",
    #     "量化交易",
    #     "今天天气怎么样？",
    #     "abcabc",
    # ]
    queries = [
        "文心一言插件怎么开发",
        "今天百度美股的股价是多少?",
        "心血管科,高血压可以蒸桑拿吗？",
        "量化交易",
        "城市景观照明中有过度照明的规定是什么？",
        "这几篇文档主要内容是什么？",
        "今天天气怎么样？",
        "abcabc",
    ]
    toolkit = RemoteToolkit.from_openapi_file("../tests/fixtures/openapi.yaml")
    for query in queries:
        memory = WholeMemory()
        if args.retrieval_type == "rag":
            agent = FunctionAgentWithRetrieval(
                llm=llm,
                knowledge_base=baizhong_db,
                top_k=3,
                threshold=0.1,
                tools=toolkit.get_tools() + [retrieval_tool],
                memory=memory,
            )
        elif args.retrieval_type == "rag_tool":
            agent = FunctionAgentWithRetrievalTool(  # type: ignore
                llm=llm,
                knowledge_base=baizhong_db,
                top_k=3,
                tools=toolkit.get_tools() + [retrieval_tool],
                memory=memory,
            )
        elif args.retrieval_type == "rag_threshold":
            agent = FunctionAgentWithRetrievalScoreTool(  # type: ignore
                llm=llm,
                knowledge_base=baizhong_db,
                top_k=3,
                threshold=0.1,
                tools=toolkit.get_tools() + [retrieval_tool],
                memory=memory,
            )
        try:
            response = asyncio.run(agent.run(query))
            print(f"query: {query}")
            print(f"agent response: {response}")
        except Exception as e:
            print(e)
