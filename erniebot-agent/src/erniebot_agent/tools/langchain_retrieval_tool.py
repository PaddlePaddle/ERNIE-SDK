from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from pydantic import Field

from erniebot_agent.memory.messages import AIMessage, HumanMessage
from erniebot_agent.tools.schema import ToolParameterView

from .base import Tool


class LangChainRetrievalToolInputView(ToolParameterView):
    query: str = Field(description="查询语句")
    top_k: int = Field(description="返回结果数量")


class SearchResponseDocument(ToolParameterView):
    title: str = Field(description="检索结果的标题")
    document: str = Field(description="检索结果的内容")


class LangChainRetrievalToolOutputView(ToolParameterView):
    documents: List[SearchResponseDocument] = Field(description="检索结果，内容和用户输入query相关的段落")


class LangChainRetrievalTool(Tool):
    description: str = "在知识库中检索与用户输入query相关的段落"
    input_type: Type[ToolParameterView] = LangChainRetrievalToolInputView
    ouptut_type: Type[ToolParameterView] = LangChainRetrievalToolOutputView

    def __init__(
        self,
        name,
        description,
        db,
        threshold: float = 0.0,
        input_type=None,
        output_type=None,
        examples=None,
        return_meta_data: bool = False,
    ) -> None:
        super().__init__()
        self.name = name
        self.db = db
        self.description = description
        self.return_meta_data = return_meta_data
        self.few_shot_examples = []
        if input_type is not None:
            self.input_type = input_type
        if output_type is not None:
            self.ouptut_type = output_type
        if examples is not None:
            self.few_shot_examples = examples
        self.threshold = threshold

    async def __call__(self, query: str, top_k: int = 3, filters: Optional[Dict[str, Any]] = None):
        documents = self.db.similarity_search_with_relevance_scores(query, top_k)
        docs = []
        for doc, score in documents:
            if score > self.threshold:
                new_doc = {"document": doc.page_content}
                if self.return_meta_data:
                    new_doc["meta"] = doc.metadata
                if "source" in doc.metadata:
                    new_doc["title"] = doc.metadata["source"]

                docs.append(new_doc)

        return {"documents": docs}

    @property
    def examples(
        self,
    ) -> List[Any]:
        few_shot_objects: List[Any] = []
        for item in self.few_shot_examples:
            few_shot_objects.append(HumanMessage(item["user"]))
            few_shot_objects.append(
                AIMessage(
                    "",
                    function_call={
                        "name": self.tool_name,
                        "thoughts": item["thoughts"],
                        "arguments": item["arguments"],
                    },
                )
            )

        return few_shot_objects
