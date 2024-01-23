import argparse
import json
import os
from typing import Any, Dict, List, Optional

import faiss
import jsonlines
from langchain.docstore.document import Document
from langchain.text_splitter import SpacyTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader
from llama_index import (
    ServiceContext,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.node_parser import SentenceSplitter
from llama_index.vector_stores.faiss import FaissVectorStore

from erniebot_agent.memory import HumanMessage, Message
from erniebot_agent.prompt import PromptTemplate
from erniebot_agent.tools.langchain_retrieval_tool import LangChainRetrievalTool
from erniebot_agent.tools.llama_index_retrieval_tool import LlamaIndexRetrievalTool

ABSTRACTPROMPT = """
{{content}} ，请用中文对上述文章进行总结。
总结需要有概括性，不允许输出与文章内容无关的信息，字数控制在500字以内
总结为：
"""


class GenerateAbstract:
    def __init__(self, llm, chunk_size: int = 1500, chunk_overlap=0, path="./abstract.json"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm = llm
        self.text_splitter = SpacyTextSplitter(
            pipeline="zh_core_web_sm", chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.prompt = PromptTemplate(ABSTRACTPROMPT, input_variables=["content"])
        self.path = path

    def data_load(self, dir_path):
        loader = DirectoryLoader(path=dir_path)
        docs = loader.load()
        return docs

    def split_documents(self, docs: List[Document]):
        return self.text_splitter.split_documents(docs)

    async def generate_abstract(self, content: str):
        content = self.prompt.format(report=content)
        messages: List[Message] = [HumanMessage(content)]
        response = await self.llm.chat(messages)
        return response.content

    async def tackle_file(self, docs: List[Document]):
        docs = self.split_documents(docs)
        summaries = []
        for doc in docs:
            summary = await self.generate_abstract(doc.page_content)
            summaries.append(summary)
        summary = "\n".join(summaries)
        if len(summaries) > 1:
            summary = await self.generate_abstract(summary)
        return summary

    def write_json(self, data: List[dict]):
        json_str = json.dumps(data, ensure_ascii=False)
        with open(self.path, "w") as json_file:
            json_file.write(json_str)

    async def run(self, data_dir, url_path=None):
        if url_path:
            _, suffix = os.path.splitext(url_path)
            assert suffix == ".json"
            with open(url_path) as f:
                url_dict = json.load(f)
        else:
            url_dict = None
        docs = self.data_load(data_dir)
        summary_list = []
        for item in docs:
            summary = await self.tackle_file([item])
            url = url_dict.get(item.metadata["source"], item.metadata["source"])
            if url_dict and item.metadata["source"] in url_dict:
                item.metadata["source"] = url_dict[item.metadata["source"]]
            summary_list.append({"page_content": summary, "url": url, "name": item.metadata["source"]})
        self.write_json(summary_list)
        return self.path


def add_url(url_dict: Dict, path: Optional[str] = None):
    if not path:
        path = "./url.json"
    json_str = json.dumps(url_dict, ensure_ascii=False)
    with open(path, "w") as json_file:
        json_file.write(json_str)


def preprocess(data_dir, url_path=None):
    loader = DirectoryLoader(path=data_dir)
    docs = loader.load()
    if url_path:
        _, suffix = os.path.splitext(url_path)
        assert suffix == ".json"
        with open(url_path) as f:
            url_dict = json.load(f)
        for item in docs:
            if "source" not in item.metadata:
                item.metadata["source"] = ""
            if item.metadata["source"] in url_dict:
                item.metadata["url"] = url_dict[item.metadata["source"]]
            else:
                item.metadata["url"] = item.metadata["source"]
    return docs


def build_index_langchain(
    faiss_name, embeddings, path=None, url_path=None, abstract=False, origin_data=None, use_data=False
):
    if os.path.exists(faiss_name):
        db = FAISS.load_local(faiss_name, embeddings)
    elif abstract and not use_data:
        all_docs = []
        with jsonlines.open(path) as reader:
            for obj in reader:
                if type(obj) is list:
                    for item in obj:
                        if "url" in item:
                            metadata = {"url": item["url"], "name": item["name"]}
                        else:
                            metadata = {"name": item["name"]}
                        doc = Document(page_content=item["page_content"], metadata=metadata)
                        all_docs.append(doc)
                elif type(obj) is dict:
                    if "url" in obj:
                        metadata = {"url": obj["url"], "name": obj["name"]}
                    else:
                        metadata = {"name": obj["name"]}
                    doc = Document(page_content=obj["page_content"], metadata=metadata)
                    all_docs.append(doc)
        db = FAISS.from_documents(all_docs, embeddings)
        db.save_local(faiss_name)
    elif not abstract and not use_data:
        documents = preprocess(path, url_path)
        text_splitter = SpacyTextSplitter(pipeline="zh_core_web_sm", chunk_size=1500, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
        docs_tackle = []
        for item in docs:
            item.metadata["name"] = item.metadata["source"].split("/")[-1].split(".")[0]
            docs_tackle.append(item)
        db = FAISS.from_documents(docs_tackle, embeddings)
        db.save_local(faiss_name)
    elif use_data:
        db = FAISS.from_documents(origin_data, embeddings)
        db.save_local(faiss_name)
    return db


def build_index_llama(
    faiss_name, embeddings, path=None, url_path=None, abstract=False, origin_data=None, use_data=False
):
    if embeddings.model == "text-embedding-ada-002":
        d = 1536
    elif embeddings.model == "ernie-text-embedding":
        d = 384
    else:
        raise ValueError(f"model {embeddings.model} not support")

    faiss_index = faiss.IndexFlatIP(d)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    if os.path.exists(faiss_name):
        vector_store = FaissVectorStore.from_persist_dir(persist_dir=faiss_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=faiss_name)
        service_context = ServiceContext.from_defaults(embed_model=embeddings)
        index = load_index_from_storage(storage_context=storage_context, service_context=service_context)
        return index
    if not abstract and not use_data:
        documents = SimpleDirectoryReader(path).load_data()
        text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        service_context = ServiceContext.from_defaults(embed_model=embeddings, text_splitter=text_splitter)
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True,
            service_context=service_context,
        )
        index.storage_context.persist(persist_dir=faiss_name)
        return storage_context


def get_retriver_by_type(frame_type):
    retriver_function = {
        "langchain": [build_index_langchain, LangChainRetrievalTool],
        "llama_index": [build_index_llama, LlamaIndexRetrievalTool],
    }
    return retriver_function[frame_type]


if __name__ == "__main__":
    import asyncio

    from langchain_openai import AzureOpenAIEmbeddings

    from erniebot_agent.extensions.langchain.embeddings import ErnieEmbeddings

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index_name_full_text",
        type=str,
        default="",
        help="The name of the full-text knowledge base(faiss)",
    )
    parser.add_argument(
        "--index_name_abstract", type=str, default="", help="The name of the abstract base(faiss)"
    )
    parser.add_argument("--path_full_text", type=str, default="", help="Full-text data storage folder path")
    parser.add_argument("--path_abstract", type=str, default="", help="json file path to store summary")
    parser.add_argument(
        "--embedding_type",
        type=str,
        default="openai_embedding",
        choices=["openai_embedding", "baizhong", "ernie_embedding"],
        help="['openai_embedding','baizhong','ernie_embedding']",
    )
    parser.add_argument("--url_path", type=str, default="", help="json file path to store url link")
    parser.add_argument(
        "--use_frame",
        type=str,
        default="langchain",
        choices=["langchain", "llama_index"],
        help="['langchain','llama_index']",
    )
    args = parser.parse_args()
    access_token = os.environ["AISTUDIO_ACCESS_TOKEN"]
    if args.embedding_type == "openai_embedding":
        embeddings: Any = AzureOpenAIEmbeddings(azure_deployment="text-embedding-ada")
    else:
        embeddings = ErnieEmbeddings(aistudio_access_token=access_token)
    _, suffix = os.path.splitext(args.url_path)
    if suffix == ".txt":
        url_path = args.url_path.replace(".txt", ".json")
        with open(args.url_path, "r") as f:
            data = f.read()
            data_list = data.split("\n")
            url_dict = {}
            for item in data_list:
                url, path = item.split(" ")
                url_dict[path] = url
            url_path
            add_url(url_dict, path=url_path)
    else:
        url_path = args.url_path
    if not args.path_abstract:
        from erniebot_agent.chat_models import ERNIEBot

        llm = ERNIEBot(model="ernie-4.0")
        generate_abstract = GenerateAbstract(llm=llm)
        abstract_path = asyncio.run(generate_abstract.run(args.path_full_text, url_path))
    else:
        abstract_path = args.path_abstract
    build_index_fuction, retrieval_tool = get_retriver_by_type(args.use_frame)
    full_text_db = build_index_fuction(
        faiss_name=args.index_name_full_text,
        embeddings=embeddings,
        path=args.path_full_text,
        url_path=url_path,
    )
    abstract_db = build_index_fuction(
        faiss_name=args.index_name_abstract,
        embeddings=embeddings,
        path=abstract_path,
        abstract=True,
        url_path=url_path,
    )
    retrieval_full = retrieval_tool(full_text_db)
    retrieval_abstract = retrieval_tool(abstract_db)
    print(asyncio.run(retrieval_full("agent的发展")))
    print(asyncio.run(retrieval_abstract("agent的发展", top_k=2)))
