import argparse
import asyncio
import os

os.environ["EB_AGENT_LOGGING_LEVEL"] = "INFO"
os.environ["EB_AGENT_ACCESS_TOKEN"] = "your access token"

from typing import List, Union

import nbformat
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.helpers import detect_file_encodings
from langchain.text_splitter import (
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.vectorstores import FAISS
from langchain_core.documents import Document
from sklearn.metrics.pairwise import cosine_similarity

from erniebot_agent.agents.function_agent_with_retrieval import (
    FunctionAgentWithRetrieval,
)
from erniebot_agent.chat_models.erniebot import ERNIEBot
from erniebot_agent.extensions.langchain.embeddings import ErnieEmbeddings
from erniebot_agent.memory import SystemMessage

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
        if cell["cell_type"] == "markdown":
            concatenated_content += cell["source"] + "\n\n"
        elif cell["cell_type"] == "code":
            concatenated_content += "```python\n" + cell["source"] + "```\n\n"

    # 返回拼接后的内容
    return concatenated_content


class NotebookLoader(BaseLoader):
    def __init__(
        self,
        file_path: str,
        encoding: str = "utf-8",
    ):
        """Initialize with file path."""
        self.file_path = file_path
        self.encoding = encoding

    def load(self) -> List[Document]:
        """Load from file path."""
        text = ""
        try:
            text = open_and_concatenate_ipynb(ipynb_path=self.file_path, encoding=self.encoding)
        except UnicodeDecodeError as e:
            if self.autodetect_encoding:
                detected_encodings = detect_file_encodings(self.file_path)
                for encoding in detected_encodings:
                    try:
                        with open(self.file_path, encoding=encoding.encoding) as f:
                            text = f.read()
                        break
                    except UnicodeDecodeError:
                        continue
            else:
                raise RuntimeError(f"Error loading {self.file_path}") from e
        except Exception as e:
            raise RuntimeError(f"Error loading {self.file_path}") from e

        metadata = {"source": self.file_path}
        return [Document(page_content=text, metadata=metadata)]


class FaissSearch:
    def __init__(self, db, embeddings):
        self.db = db
        self.embeddings = embeddings

    def search(self, query: str, top_k: int = 10, **kwargs):
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
        return retrieval_results


def load_agent():
    faiss_name = "faiss_index"
    if not args.init:
        db = FAISS.load_local(faiss_name, embeddings)
    else:
        md_file_path = [
            "./docs/modules/file.md",
            "./docs/modules/agents.md",
            "docs/modules/memory.md",
            "./docs/modules/message.md",
            "./docs/modules/chat_models.md",
        ]
        content = read_md_files(md_file_path)
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        md_header_splits = markdown_splitter.split_text(content)
        chunk_size = 500
        chunk_overlap = 30
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        splits = text_splitter.split_documents(md_header_splits)
        db = FAISS.from_documents(splits, embeddings)

        ipynb_path = [
            "./docs/cookbooks/agent/function_agent.ipynb",
            "./docs/cookbooks/agent/chat_models.ipynb",
            "./docs/cookbooks/agent/memory.ipynb",
            "./docs/cookbooks/agent/message.ipynb",
        ]
        for notebook in ipynb_path:
            concatenated_content = open_and_concatenate_ipynb(notebook, encoding="utf-8")
            text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
            db.add_documents(text_splitter.create_documents([concatenated_content]))
        db.save_local(faiss_name)

    llm = ERNIEBot(model="ernie-3.5")
    faiss_search = FaissSearch(db=db, embeddings=embeddings)
    agent = FunctionAgentWithRetrieval(
        llm=llm,
        tools=[],
        knowledge_base=faiss_search,
        threshold=0,
        system_message=SystemMessage(
            "你是ERNIEBot Agent的小助手，用于解决用户关于EB-Agent的问题，"
            "涉及File, Memory, Message, Agent, ChatModels等模块。请你严格按照搜索到的内容回答，不要自己生成相关代码。"
        ),
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
