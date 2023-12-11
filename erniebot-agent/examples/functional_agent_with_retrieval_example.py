import argparse
import asyncio

from erniebot_agent.agents import FunctionalAgentWithRetrievalScoreTool
from erniebot_agent.chat_models import ERNIEBot
from erniebot_agent.memory import WholeMemory
from erniebot_agent.retrieval import BaizhongSearch
from erniebot_agent.retrieval.document import Document
from erniebot_agent.tools.baizhong_tool import BaizhongSearchTool
from erniebot_agent.tools.base import RemoteToolkit
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import SpacyTextSplitter
from tqdm import tqdm

import erniebot

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
        base_url=args.base_url,
        project_name="construct_assistant2",
        remark="construction assistant test dataset",
        project_id=args.project_id if args.project_id != -1 else None,
    )
    print(baizhong_db.project_id)
    if args.indexing:
        res = offline_ann(args.data_path, baizhong_db)
        print(res)

    llm = ERNIEBot(model="ernie-bot", api_type="custom")

    retrieval_tool = BaizhongSearchTool(
        description="Use Baizhong Search to retrieve documents.", db=baizhong_db, threshold=0.1
    )

    # agent = FunctionalAgentWithRetrievalTool(
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
        "量化交易",
        "城市景观照明中有过度照明的规定是什么？",
        "今天天气怎么样？",
        "abcabc",
    ]
    toolkit = RemoteToolkit.from_openapi_file("../tests/fixtures/openapi.yaml")
    for query in queries:
        memory = WholeMemory()
        agent = FunctionalAgentWithRetrievalScoreTool(
            llm=llm,
            knowledge_base=baizhong_db,
            top_k=3,
            threshold=0.1,
            tools=toolkit.get_tools() + [retrieval_tool],
            memory=memory,
        )
        response = asyncio.run(agent.async_run(query))
        print(f"query: {query}")
        print(f"agent response: {response}")
