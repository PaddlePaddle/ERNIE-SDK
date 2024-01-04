import json
import logging
import os
import shutil
import urllib.parse
from typing import Optional

import erniebot
import jsonlines
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import SpacyTextSplitter
from langchain.vectorstores import FAISS
from md2pdf.core import md2pdf
from sklearn.metrics.pairwise import cosine_similarity

from erniebot_agent.agents.base import BaseAgent
from erniebot_agent.agents.callback import CallbackHandler
from erniebot_agent.prompt import PromptTemplate

default_logger = logging.getLogger(__name__)


class ReportCallbackHandler(CallbackHandler):
    logger: logging.Logger

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize a logging handler.

        Args:
            logger: The logger to use. If `None`, a default logger will be used.
        """
        super().__init__()

        if logger is None:
            self.logger = default_logger
        else:
            self.logger = logger

    async def on_run_start(self, agent: BaseAgent, prompt, **kwargs):
        agent_name = kwargs.get("agent_name", None)
        self.logger.info(f"{agent_name}开始运行：{prompt}")

    async def on_run_end(self, agent: BaseAgent, response, **kwargs):
        agent_name = kwargs.get("agent_name", None)
        self.logger.info(f"{agent_name}结束运行,{response}")

    async def on_run_tool(self, tool_name, response):
        self.logger.info(f"{tool_name}的运行结果：{response}")

    async def on_run_error(self, tool_name, error_information):
        self.logger.error(f"{tool_name}的调用失败，错误信息：{error_information}")


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
            retrieval_results.append(
                {
                    "content": doc.page_content,
                    "score": similarities[index],
                    "title": doc.metadata["name"],
                    "url": doc.metadata["url"],
                }
            )
        return retrieval_results


def build_index(faiss_name, embeddings, path=None, abstract=False, origin_data=None, use_data=False):
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
        loader = PyPDFDirectoryLoader(path)
        documents = loader.load()
        text_splitter = SpacyTextSplitter(pipeline="zh_core_web_sm", chunk_size=1500, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
        docs_tackle = []
        for item in docs:
            item.metadata["name"] = item.metadata["source"].split("/")[-1].replace(".pdf", "")
            item.metadata["url"] = item.metadata["source"]
            docs_tackle.append(item)
        db = FAISS.from_documents(docs_tackle, embeddings)
        db.save_local(faiss_name)
    elif use_data:
        db = FAISS.from_documents(origin_data, embeddings)
        db.save_local(faiss_name)
    return db


def erniebot_chat(messages: list, functions: Optional[str] = None, model: Optional[str] = None, **kwargs):
    if not model:
        model = "ernie-4.0"
    if functions is None:
        resp_stream = erniebot.ChatCompletion.create(model=model, messages=messages, **kwargs, stream=False)
    else:
        resp_stream = erniebot.ChatCompletion.create(
            model=model, messages=messages, **kwargs, functions=functions, stream=False
        )
    return resp_stream["result"]


def write_to_file(filename: str, text: str) -> None:
    """Write text to a file

    Args:
        text (str): The text to write
        filename (str): The filename to write to
    """
    with open(filename, "w") as file:
        file.write(text)


def md_to_pdf(input_file, output_file):
    md2pdf(output_file, md_content=None, md_file_path=input_file, css_file_path=None, base_url=None)


def write_md_to_pdf(task: str, path: str, text: str) -> str:
    file_path = f"{path}/{task}"
    write_to_file(f"{file_path}.md", text)

    # encoded_file_path = urllib.parse.quote(f"{file_path}.pdf")
    encoded_file_path = urllib.parse.quote(f"{file_path}.md")
    return encoded_file_path


def write_to_json(filename: str, list_data: list, mode="w") -> None:
    """Write text to a file

    Args:
        text (str): The text to write
        filename (str): The filename to write to
    """
    with jsonlines.open(filename, mode) as file:
        for item in list_data:
            file.write(item)


def add_citation(paragraphs, faiss_name, embeddings):
    if os.path.exists(faiss_name):
        shutil.rmtree(faiss_name)
    list_data = []
    for item in paragraphs:
        example = Document(page_content=item["summary"], metadata={"url": item["url"], "name": item["name"]})
        list_data.append(example)
    faiss_db = build_index(
        faiss_name=faiss_name, use_data=True, embeddings=embeddings, origin_data=list_data
    )
    faiss_search = FaissSearch(db=faiss_db, embeddings=embeddings)
    return faiss_search


def postprocess(report):
    prompt_abstract = """
    请你总结报告并给出报告的摘要和关键词，摘要在100-200字之间，关键词不超过5个词。
    你需要输出一个json形式的字符串，内容为{'摘要':...,'关键词':...}。
    现在给你报告的内容：
    {{report}}"""
    polish_prompt = """你的任务是扩写和润色相关内容，
    你需要把相关内容扩写到300-400字之间，扩写的内容必须与给出的内容相关。
    下面给出内容:
    {content}
    扩写并润色内容为:"""
    Prompt_abstract = PromptTemplate(prompt_abstract, input_variables=["report"])
    messages = [{"role": "user", "content": Prompt_abstract.format(report=report)}]
    if len(messages[0]["content"]) > 4800:
        model = "ernie-longtext"
    else:
        model = "ernie-4.0"
    while True:
        try:
            abstract_json = erniebot_chat(messages, model=model)
            l_index = abstract_json.find("{")
            r_index = abstract_json.rfind("}")
            abstract_json = json.loads(abstract_json[l_index : r_index + 1])
            abstract = abstract_json["摘要"]
            key = abstract_json["关键词"]
            if type(key) is list:
                key = "，".join(key)
            break
        except Exception as e:
            print(e)
            continue
    report_list = report.split("\n\n")
    if "#" in report_list[0] and "##" in report_list[1]:
        paragraphs = []
        title = report_list[0]
        paragraphs.append(title)
        paragraphs.append("**摘要** " + abstract)
        paragraphs.append("**关键词** " + key)
        content = ""
        for item in report_list[1:]:
            if "#" not in item:
                content += item + "\n"
            else:
                if len(content) > 300:
                    paragraphs.append(content)
                elif len(content) > 0:
                    messages = [{"role": "user", "content": polish_prompt.format(content=content)}]
                    paragraphs.append(erniebot_chat(messages=messages))
                content = ""
                paragraphs.append(item)
        return "\n\n".join(paragraphs)
    else:
        raise Exception("Report format error")
