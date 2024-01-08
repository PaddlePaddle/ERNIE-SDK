from __future__ import annotations

from functools import partial

from langchain.chains.combine_documents import collapse_docs, split_list_of_docs
from langchain.prompts import PromptTemplate
from langchain.schema import Document, StrOutputParser
from langchain.schema.prompt_template import format_document
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.text_splitter import SpacyTextSplitter

from erniebot_agent.extensions.langchain.llms import ErnieBot
from erniebot_agent.tools.base import Tool

TOKEN_MAX_LENGTH = 4800


class TextSummarizationTool(Tool):
    description: str = "text summarization tool"

    def map_reduce(self, question: str = ""):
        llm = ErnieBot()
        document_prompt = PromptTemplate.from_template("{page_content}")
        partial_format_document = partial(format_document, prompt=document_prompt)
        prompt = (
            "根据给出的文本内容，简要回答以下问题："
            + question
            + '—— 如果无法使用文本回答问题，请简要总结文本。"包括所有的事实信息、数字、统计数据等（如果有的话）。字数控制在350字以内。文本内容：\n\n{context}'
        )
        map_chain = (
            {"context": partial_format_document}
            | PromptTemplate.from_template(prompt)
            | llm
            | StrOutputParser()
        )
        map_as_doc_chain = (
            RunnableParallel({"doc": RunnablePassthrough(), "content": map_chain})
            | (lambda x: Document(page_content=x["content"], metadata=x["doc"].metadata))
        ).with_config(run_name="Summarize (return doc)")

        def format_docs(docs):
            return "\n\n".join(partial_format_document(doc) for doc in docs)

        collapse_chain = (
            {"context": format_docs} | PromptTemplate.from_template(prompt) | llm | StrOutputParser()
        )

        def get_num_tokens(docs):
            return len(docs)

        def collapse(
            docs,
            config,
            token_max=4000,
        ):
            collapse_ct = 1
            while len(docs) > token_max:
                config["run_name"] = f"Collapse {collapse_ct}"
                invoke = partial(collapse_chain.invoke, config=config)
                split_docs = split_list_of_docs(docs, get_num_tokens, token_max)
                docs = [collapse_docs(_docs, invoke) for _docs in split_docs]
                collapse_ct += 1
            return docs

        reduce_chain = (
            {"context": format_docs}
            | PromptTemplate.from_template("合并这些总结:\n\n{context}")
            | llm
            | StrOutputParser()
        ).with_config(run_name="Reduce")
        map_reduce = (map_as_doc_chain.map() | collapse | reduce_chain).with_config(run_name="Map reduce")
        return map_reduce

    async def __call__(
        self,
        text: str,
        question: str,
        **kwargs,
    ):
        if not text:
            return "Error: No text to summarize"
        if len(text) > TOKEN_MAX_LENGTH - 500:
            docs = [
                Document(
                    page_content=text,
                    metadata={},
                )
            ]
            text_splitter = SpacyTextSplitter(pipeline="zh_core_web_sm", chunk_size=1500, chunk_overlap=0)
            docs = text_splitter.split_documents(docs)
        else:
            docs = [
                Document(
                    page_content=text,
                    metadata={},
                )
            ]
        map_reduce = self.map_reduce(question)
        summary = map_reduce.invoke(docs, config={"max_concurrency": 1})
        return summary
