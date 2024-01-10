from typing import Any, Dict, List, Optional

from pydantic import Field

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

    def __init__(
        self,
        db,
        threshold: float = 0.0,
        input_type=None,
        output_type=None,
        return_meta_data: bool = True,
    ) -> None:
        super().__init__()
        self.db = db
        self.return_meta_data = return_meta_data
        self.few_shot_examples = []
        if input_type is not None:
            self.input_type = input_type
        if output_type is not None:
            self.ouptut_type = output_type
        self.threshold = threshold

    async def __call__(self, query: str, top_k: int = 3, filters: Optional[Dict[str, Any]] = None):
        documents = self.db.similarity_search_with_relevance_scores(query, top_k)
        docs = []
        for doc, score in documents:
            if score > self.threshold:
                new_doc = {"content": doc.page_content, "score": score}
                if self.return_meta_data:
                    new_doc["meta"] = doc.metadata
                docs.append(new_doc)

        return {"documents": docs}
