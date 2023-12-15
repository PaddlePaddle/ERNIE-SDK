import argparse
import asyncio

from erniebot_agent.agents import FunctionalAgentWithQueryPlanning
from erniebot_agent.chat_models import ERNIEBot
from erniebot_agent.memory import WholeMemory
from erniebot_agent.retrieval import BaizhongSearch
from erniebot_agent.tools.baizhong_tool import BaizhongSearchTool
from erniebot_agent.tools.base import RemoteToolkit

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
parser.add_argument(
    "--retrieval_type",
    choices=["summary_fulltext_tools", "knowledge_tools"],
    default="knowledge_tools",
    help="Retrieval type, default to rag.",
)
args = parser.parse_args()

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

    llm = ERNIEBot(model="ernie-bot", api_type="custom")

    # 建筑规范数据集
    retrieval_tool = BaizhongSearchTool(
        name="construction_search", description="提供城市管理执法办法相关的信息", db=baizhong_db, threshold=0.1
    )
    # OpenAI数据集
    openai_tool = BaizhongSearchTool(
        name="openai_search", description="提供关于OpenAI公司的信息", db=baizhong_db, threshold=0.1
    )
    # 金融数据集
    finance_tool = BaizhongSearchTool(
        name="financial_search", description="提供关于量化交易相关的信息", db=baizhong_db, threshold=0.1
    )

    summary_tool = BaizhongSearchTool(
        name="text_summary_search", description="使用这个工具总结与作者生活相关的问题", db=baizhong_db, threshold=0.1
    )
    vector_tool = BaizhongSearchTool(
        name="fulltext_search", description="使用这个工具检索特定的上下文，以回答有关作者生活的特定问题", db=baizhong_db, threshold=0.1
    )
    queries = [
        "量化交易",
        "OpenAI管理层变更会带来哪些影响?" "城市景观照明中有过度照明的规定是什么？",
        "城市景观照明中有过度照明的规定是什么？并把搜索的内容添加到笔记本中",
        "这几篇文档主要内容是什么？",
        "今天天气怎么样？",
        "abcabc",
    ]
    toolkit = RemoteToolkit.from_openapi_file("../tests/fixtures/openapi.yaml")
    for query in queries:
        memory = WholeMemory()
        if args.retrieval_type == "summary_fulltext":
            agent = FunctionalAgentWithQueryPlanning(  # type: ignore
                llm=llm,
                knowledge_base=baizhong_db,
                top_k=3,
                tools=[summary_tool, vector_tool],
                memory=memory,
            )
        elif args.retrieval_type == "knowledge_tools":
            agent = FunctionalAgentWithQueryPlanning(  # type: ignore
                llm=llm,
                knowledge_base=baizhong_db,
                top_k=3,
                tools=toolkit.get_tools() + [retrieval_tool, openai_tool, finance_tool],
                memory=memory,
            )

        try:
            response = asyncio.run(agent.async_run(query))
            print(f"query: {query}")
            print(f"agent response: {response}")
        except Exception as e:
            print(e)
