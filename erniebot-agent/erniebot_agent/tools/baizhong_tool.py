from __future__ import annotations

from typing import Any, List, Optional, Type

from erniebot_agent.messages import AIMessage, HumanMessage
from erniebot_agent.tools.schema import ToolParameterView
from pydantic import Field

from .base import Tool


class BaizhongSearchToolInputView(ToolParameterView):
    query: str = Field(description="查询语句")
    top_k: int = Field(description="返回结果数量")


class SearchResponseDocument(ToolParameterView):
    id: str = Field(description="检索结果的文本的id")
    title: str = Field(description="检索结果的标题")
    document: str = Field(description="检索结果的内容")


class BaizhongSearchToolOutputView(ToolParameterView):
    documents: List[SearchResponseDocument] = Field(description="检索结果，内容和用户输入query相关的段落")


class BaizhongSearchTool(Tool):
    description: str = "在知识库中检索与用户输入query相关的段落"
    input_type: Type[ToolParameterView] = BaizhongSearchToolInputView
    ouptut_type: Type[ToolParameterView] = BaizhongSearchToolOutputView

    def __init__(
        self, description, db, threshold: float = 0.0, input_type=None, output_type=None, examples=None
    ) -> None:
        super().__init__()
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

    async def __call__(self, query: str, top_k: int = 3, filters: Optional[dict[str, Any]] = None):
        documents = self.db.search(query, top_k, filters)
        documents = [item for item in documents if item["score"] > self.threshold]
        return {"documents": documents}

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
