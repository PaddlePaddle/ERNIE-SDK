import argparse
import asyncio
from typing import Dict, List, Type

from erniebot_agent.agents import FunctionalAgentWithQueryPlanning
from erniebot_agent.chat_models import ERNIEBot
from erniebot_agent.memory import WholeMemory
from erniebot_agent.messages import AIMessage, HumanMessage, Message
from erniebot_agent.retrieval import BaizhongSearch
from erniebot_agent.tools.baizhong_tool import BaizhongSearchTool
from erniebot_agent.tools.base import RemoteToolkit, Tool
from erniebot_agent.tools.openai_search_tool import OpenAISearchTool
from erniebot_agent.tools.schema import ToolParameterView
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from pydantic import Field

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
parser.add_argument(
    "--search_engine",
    choices=["baizhong", "openai"],
    default="baizhong",
    help="search_engine.",
)
args = parser.parse_args()


class NotesToolInputView(ToolParameterView):
    draft: str = Field(description="草稿文本")


class NotesToolOutputView(ToolParameterView):
    draft_results: str = Field(description="草稿文本结果")


class NotesTool(Tool):
    description: str = "笔记本，用于记录和保存信息的笔记本工具"
    input_type: Type[ToolParameterView] = NotesToolInputView
    ouptut_type: Type[ToolParameterView] = NotesToolOutputView

    async def __call__(self, draft: str) -> Dict[str, str]:
        # TODO: save draft to database
        return {"draft_results": "草稿在笔记本中保存成功"}

    @property
    def examples(self) -> List[Message]:
        return [
            HumanMessage("OpenAI管理层变更会带来哪些影响？并请把搜索的内容添加到笔记本中"),
            AIMessage(
                "",
                function_call={
                    "name": self.tool_name,
                    "thoughts": f"用户想保存笔记，我可以使用{self.tool_name}工具来保存，其中`draft`字段的内容为：'搜索的草稿'。",
                    "arguments": '{"draft": "搜索的草稿"}',
                },
            ),
        ]


if __name__ == "__main__":
    erniebot.api_type = args.api_type
    erniebot.access_token = args.access_token

    llm = ERNIEBot(model="ernie-bot", api_type="custom")
    if args.search_engine == "baizhong":
        baizhong_db = BaizhongSearch(
            base_url=args.base_url,
            project_name="construct_assistant2",
            remark="construction assistant test dataset",
            project_id=args.project_id if args.project_id != -1 else None,
        )
        print(baizhong_db.project_id)
        # 建筑规范数据集
        city_management = BaizhongSearchTool(
            name="city_administrative_law_enforcement",
            description="提供城市管理执法办法相关的信息",
            db=baizhong_db,
            threshold=0.1,
        )
        city_design = BaizhongSearchTool(
            name="city_design_management", description="提供城市设计管理办法的信息", db=baizhong_db, threshold=0.1
        )
        city_lighting = BaizhongSearchTool(
            name="city_lighting", description="提供关于城市照明管理规定的信息", db=baizhong_db, threshold=0.1
        )

        tool_retriever = BaizhongSearchTool(
            name="tool_retriever", description="用于检索与query相关的tools列表", db=baizhong_db, threshold=0.1
        )
    elif args.search_engine == "openai" and args.retrieval_type == "knowledge_tools":
        embeddings = OpenAIEmbeddings(deployment="text-embedding-ada")
        faiss = FAISS.load_local("城市管理执法办法", embeddings)
        openai_city_management = OpenAISearchTool(
            name="city_administrative_law_enforcement",
            description="提供城市管理执法办法相关的信息",
            db=faiss,
            threshold=0.1,
        )
        faiss = FAISS.load_local("城市设计管理办法", embeddings)
        openai_city_design = OpenAISearchTool(
            name="city_design_management", description="提供城市设计管理办法的信息", db=faiss, threshold=0.1
        )
        faiss = FAISS.load_local("城市照明管理规定", embeddings)
        openai_city_lighting = OpenAISearchTool(
            name="city_lighting", description="提供关于城市照明管理规定的信息", db=faiss, threshold=0.1
        )
        # TODO(wugaoshewng) 加入APE后，变成knowledge_base_toolkit
        # faiss = FAISS.load_local("tool_retriever", embeddings)
        tool_map = {
            "city_administrative_law_enforcement": openai_city_management,
            "city_design_management": openai_city_design,
            "city_lighting": openai_city_lighting,
        }
        docs = []
        for tool in tool_map.values():
            doc = Document(page_content=tool.description, metadata={"tool_name": tool.name})
            docs.append(doc)

        faiss_tool = FAISS.from_documents(docs, embeddings)
        tool_retriever = OpenAISearchTool(  # type: ignore
            name="tool_retriever",
            description="用于检索与query相关的tools列表",
            db=faiss_tool,
            threshold=0.1,
            return_meta_data=True,
        )
    elif args.search_engine == "openai" and args.retrieval_type == "summary_fulltext_tools":
        embeddings = OpenAIEmbeddings(deployment="text-embedding-ada")
        summary_faiss = FAISS.load_local("summary", embeddings)
        summary_tool = OpenAISearchTool(
            name="text_summary_search", description="使用这个工具总结与建筑规范相关的问题", db=summary_faiss, threshold=0.1
        )
        fulltext_faiss = FAISS.load_local("fulltext", embeddings)
        vector_tool = OpenAISearchTool(
            name="fulltext_search",
            description="使用这个工具检索特定的上下文，以回答有关建筑规范具体的问题",
            db=fulltext_faiss,
            threshold=0.1,
        )

    queries = [
        "量化交易",
        "OpenAI管理层变更会带来哪些影响?",
        "城市景观照明中有过度照明的规定是什么？",
        "城市景观照明中有过度照明的规定是什么？并把搜索的内容添加到笔记本中",
        "请比较一下城市设计管理和照明管理规定的区别？",
        "这几篇文档主要内容是什么？",
        "今天天气怎么样？",
        "abcabc",
    ]
    toolkit = RemoteToolkit.from_openapi_file("../tests/fixtures/openapi.yaml")
    for query in queries:
        memory = WholeMemory()
        if args.retrieval_type == "summary_fulltext_tools":
            agent = FunctionalAgentWithQueryPlanning(  # type: ignore
                llm=llm,
                top_k=3,
                tools=[summary_tool, vector_tool],
                memory=memory,
            )
        elif args.retrieval_type == "knowledge_tools":
            tool_results = asyncio.run(tool_retriever(query))["documents"]
            selected_tools = [tool_map[item["meta"]["tool_name"]] for item in tool_results]
            agent = FunctionalAgentWithQueryPlanning(  # type: ignore
                llm=llm,
                top_k=3,
                # tools=toolkit.get_tools() + [city_management, city_design, city_lighting],
                # tools = [NotesTool(),city_management, city_design, city_lighting],
                tools=[NotesTool()] + selected_tools,
                memory=memory,
            )

        try:
            response = asyncio.run(agent.async_run(query))
            print(f"query: {query}")
            print(f"agent response: {response}")
        except Exception as e:
            print(e)
