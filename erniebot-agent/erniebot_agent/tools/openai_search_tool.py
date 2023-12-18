from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from erniebot_agent.messages import AIMessage, HumanMessage
from erniebot_agent.tools.schema import ToolParameterView
from pydantic import Field

from .base import Tool


class OpenAISearchToolInputView(ToolParameterView):
    query: str = Field(description="查询语句")
    top_k: int = Field(description="返回结果数量")


class SearchResponseDocument(ToolParameterView):
    title: str = Field(description="检索结果的标题")
    document: str = Field(description="检索结果的内容")


class OpenAISearchToolOutputView(ToolParameterView):
    documents: List[SearchResponseDocument] = Field(description="检索结果，内容和用户输入query相关的段落")


class OpenAISearchTool(Tool):
    description: str = "在知识库中检索与用户输入query相关的段落"
    input_type: Type[ToolParameterView] = OpenAISearchToolInputView
    ouptut_type: Type[ToolParameterView] = OpenAISearchToolOutputView

    def __init__(
        self, name, description, db, threshold: float = 0.0, input_type=None, output_type=None, examples=None
    ) -> None:
        super().__init__()
        self.name = name
        self.db = db
        self.description = description
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
                docs.append(
                    {"document": doc.page_content, "title": doc.metadata["source"], "meta": doc.metadata}
                )

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
