import argparse
import asyncio
from typing import List

import erniebot
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import SpacyTextSplitter
from tqdm import tqdm

from erniebot_agent.agents.functional_agent import FunctionalAgent
from erniebot_agent.chat_models.erniebot import ERNIEBot
from erniebot_agent.memory.whole_memory import WholeMemory
from erniebot_agent.retrieval.baizhong_search import BaizhongSearch
from erniebot_agent.retrieval.document import Document
from erniebot_agent.tools.baizhong_tool import (
    BaizhongSearchTool,
    BaizhongSearchToolInputView,
    BaizhongSearchToolOutputView,
    SearchResponseDocument,
)

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default="construction_regulations", type=str, help="The data path.")
parser.add_argument(
    "--access_token", default="ai_studio_access_token", type=str, help="The aistudio access token."
)
parser.add_argument("--api_type", default="qianfan", type=str, help="The aistudio access token.")
parser.add_argument("--api_key", default="", type=str, help="The API Key.")
parser.add_argument("--secret_key", default="", type=str, help="The secret key.")
parser.add_argument("--indexing", action="store_true", help="The indexing step.")
parser.add_argument("--knowledge_base_id", default="", type=str, help="The knowledge base id.")
parser.add_argument(
    "--knowledge_base_name", default="knowledge_base_name", type=str, help="The knowledge base name."
)

args = parser.parse_args()


def offline_ann(data_path, aurora_db):
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

    res = aurora_db.add_documents(documents=list_data)
    return res


if __name__ == "__main__":
    aurora_db = BaizhongSearch(
        knowledge_base_name=args.knowledge_base_name,
        access_token=args.access_token,
        knowledge_base_id=args.knowledge_base_id if args.knowledge_base_id != "" else None,
    )
    if args.indexing:
        offline_ann(args.data_path, aurora_db)

    query = "城乡建设部规章中描述的城市管理执法的执法主体是谁？"
    # Doc store test
    result = aurora_db.search(query=query, top_k=3, filters=None)
    print(result)
    # Field description
    field_map = {
        "query": {"type": str, "description": "查询语句"},
        "top_k": {"type": int, "description": "返回结果数量"},
    }
    input_view = BaizhongSearchToolInputView.from_dict(field_map=field_map)

    field_map = {
        "id": {"type": str, "description": "规章文本的id"},
        "title": {"type": str, "description": "规章的标题"},
        "document": {"type": str, "description": "规章的内容"},
    }

    respone_view_type = SearchResponseDocument.from_dict(field_map=field_map)
    field_map = {
        "documents": {
            "type": List[respone_view_type],  # type: ignore
            "description": "检索结果，内容为住房和城乡建设部规章中和query相关的规章片段",
        }
    }
    output_view = BaizhongSearchToolOutputView.from_dict(field_map=field_map)
    print(input_view.function_call_schema())
    print(output_view.function_call_schema())

    if args.api_type == "aistudio":
        erniebot.api_type = "aistudio"
        erniebot.access_token = args.access_token
    elif args.api_type == "qianfan":
        erniebot.api_type = "qianfan"
        erniebot.ak = args.api_key
        erniebot.sk = args.secret_key

    # Few shot examples
    few_shot_examples = [
        {
            "user": "城乡建设部规章中描述的城市管理执法的执法主体是谁？",
            "thoughts": "这是一个住房和城乡建设部规章的问题，我们使用BaizhongSearchTool工具检索相关的信息，检索的query：'城市管理执法的执法主体'}",
            "arguments": '{"query": "城市管理执法的执法主体", "top_k": 3}',
        }
    ]

    aurora_search = BaizhongSearchTool(
        description="在住房和城乡建设部规章中寻找和query最相关的片段",
        db=aurora_db,
        input_type=input_view,
        output_type=output_view,
        examples=few_shot_examples,
    )
    print(aurora_search.function_call_schema())
    # Tool Test
    result = asyncio.run(aurora_search(query=query))
    llm = ERNIEBot(model="ernie-bot", api_type="custom")
    memory = WholeMemory()
    # Agent test
    agent = FunctionalAgent(llm=llm, tools=[aurora_search], memory=memory)
    response = asyncio.run(agent.async_run(query))
    print(response)
