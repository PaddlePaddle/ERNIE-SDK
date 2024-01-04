import argparse
import asyncio
import os

os.environ["EB_AGENT_LOGGING_LEVEL"] = "INFO"
os.environ["EB_AGENT_ACCESS_TOKEN"] = "your access token"

from typing import List, Union

import nbformat
from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.vectorstores import FAISS
from sklearn.metrics.pairwise import cosine_similarity

from erniebot_agent.agents.function_agent_with_retrieval import (
    FunctionAgentWithRetrieval,
)
from erniebot_agent.chat_models import ERNIEBot
from erniebot_agent.extensions.langchain.embeddings import ErnieEmbeddings
from erniebot_agent.memory import LimitTokensMemory, SystemMessage

parser = argparse.ArgumentParser()
parser.add_argument("--init", type=bool, default=False)
args = parser.parse_args()

embeddings = ErnieEmbeddings(aistudio_access_token=os.environ["EB_AGENT_ACCESS_TOKEN"], chunk_size=16)

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    # ("###", "Header 3"),
    # ("####", "Header 4"),
]


def read_md_file(file_path: str) -> Union[str, None]:
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            md_content = file.read()
        return md_content
    except FileNotFoundError:
        print(f"文件 '{file_path}' 不存在。")
        return None
    except Exception as e:
        print(f"读取文件时出现错误： {e}")
        return None


def read_md_files(file_paths: Union[str, List[str]]) -> Union[List[str], str]:
    if isinstance(file_paths, str):
        return read_md_file(file_paths)
    elif isinstance(file_paths, list):
        md_contents = ""
        for file_path in file_paths:
            content = read_md_file(file_path)
            if content is not None:
                md_contents += content
        return md_contents


def open_and_concatenate_ipynb(ipynb_path, encoding):
    # 读取.ipynb文件
    with open(ipynb_path, "r", encoding=encoding) as f:
        notebook_content = nbformat.read(f, as_version=4)

    # 按顺序拼接Markdown文本和code单元
    concatenated_content = ""
    for cell in notebook_content["cells"]:
        # if cell["cell_type"] == "markdown":
        #     concatenated_content += cell["source"] + "\n\n"
        if cell["cell_type"] == "code":
            concatenated_content += "```python\n" + cell["source"] + "```\n\n"

    # 返回拼接后的内容
    return concatenated_content


class FaissSearch:
    def __init__(self, db, embeddings, module_db):
        self.db = db
        self.module_db = module_db
        self.embeddings = embeddings

    def search(self, query: str, top_k: int = 2, **kwargs):
        docs = self.db.similarity_search(query, top_k)
        para_result = self.embeddings.embed_documents([i.page_content for i in docs])
        query_result = self.embeddings.embed_query(query)
        similarities = cosine_similarity([query_result], para_result).reshape((-1,))
        retrieval_results = []
        for index, doc in enumerate(docs):
            if "Header 1" in doc.metadata:
                retrieval_results.append(
                    {
                        "content": doc.page_content,
                        "score": similarities[index],
                        "title": doc.metadata["Header 1"],
                    }
                )
            else:
                retrieval_results.append(
                    {"content": doc.page_content, "score": similarities[index], "title": ""}
                )
        code = self.module_db.similarity_search(query, 1)[0]
        retrieval_results.append({"content": code.metadata["ipynb"], "score": 1, "title": code.page_content})

        return retrieval_results


def load_agent():
    faiss_name = "faiss_index"
    faiss_name_module = "faiss_index_module"
    if not args.init:
        db = FAISS.load_local(faiss_name, embeddings)
        module_db = FAISS.load_local(faiss_name_module, embeddings)
    else:
        md_file_path = [
            "./docs/modules/file.md",
            "./docs/modules/agents.md",
            "docs/modules/memory.md",
            "./docs/modules/message.md",
            "./docs/modules/chat_models.md",
            "./docs/modules/tools.md",
        ]
        content = read_md_files(md_file_path)
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        md_header_splits = markdown_splitter.split_text(content)
        chunk_size = 500
        chunk_overlap = 30
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        splits = text_splitter.split_documents(md_header_splits)
        db = FAISS.from_documents(splits, embeddings)
        db.save_local(faiss_name)

        ipynb_path = [
            "./docs/cookbooks/agent/function_agent.ipynb",
            "./docs/cookbooks/agent/chat_models.ipynb",
            "./docs/cookbooks/agent/memory.ipynb",
            "./docs/cookbooks/agent/message.ipynb",
            "./docs/cookbooks/agent/local_tool.ipynb",
            "./docs/cookbooks/agent/tools_intro.ipynb",
        ]
        modules = [item[item.rfind("/") + 1 : item.rfind(".ipynb")] for item in ipynb_path]
        module_doc = []
        from langchain_core.documents import Document

        for i in range(len(modules)):
            module_doc.append(
                Document(
                    page_content=modules[i],
                    metadata={"ipynb": open_and_concatenate_ipynb(ipynb_path[i], "utf-8")},
                )
            )

        module_db = FAISS.from_documents(module_doc, embeddings)
        module_db.save_local(faiss_name_module)

    llm = ERNIEBot(model="ernie-3.5")
    faiss_search = FaissSearch(db=db, embeddings=embeddings, module_db=module_db)
    memory = LimitTokensMemory(max_token_limit=4000)
    agent = FunctionAgentWithRetrieval(
        llm=llm,
        tools=[],
        knowledge_base=faiss_search,
        threshold=0,
        system_message=SystemMessage(
            "你是ERNIEBot Agent的小助手，用于解决用户关于EB-Agent的问题，"
            "涉及File, Memory, Message, Agent, ChatModels等模块。请你严格按照搜索到的内容回答，不要自己生成相关代码。"
        ),
        top_k=2,
        token_limit=5000,
        memory=memory,
    )
    return agent


async def main(agent):
    # response = await agent.run('怎么使用File模块？')
    # response = await agent.run('怎么获取我的File的内容？')
    response = await agent.run("怎么创建一个EB-Agent，给出具体的代码？")
    print(response.text)


if __name__ == "__main__":
    agent = load_agent()
    asyncio.run(main(agent))
    # agent.launch_gradio_demo()
