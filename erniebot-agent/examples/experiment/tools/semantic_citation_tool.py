from __future__ import annotations

import string
from typing import List, Type

import numpy as np
from erniebot_agent.tools.base import Tool
from erniebot_agent.tools.schema import ToolParameterView
from pydantic import Field
from sklearn.metrics.pairwise import cosine_similarity
from utils import embeddings


class SemanticCitationToolInputView(ToolParameterView):
    query: str = Field(description="Chunk of text to summarize")


class SemanticCitationToolOutputView(ToolParameterView):
    document: str = Field(description="content")


class SemanticCitationTool(Tool):
    description: str = "semantic citation tool"
    input_type: Type[ToolParameterView] = SemanticCitationToolInputView
    ouptut_type: Type[ToolParameterView] = SemanticCitationToolOutputView

    def is_punctuation(self, char: str):
        """判断一个字符是否是标点符号"""
        return char in string.punctuation

    async def __call__(self, reports: str, paragraphs: List[dict], url_index: dict):
        list_data = reports.split("\n\n")
        documents = [item["summary"] for item in paragraphs]
        para_result = embeddings.embed_documents(documents)
        output_text = []
        for chunk_text in list_data:
            if "参考文献" in chunk_text:
                output_text.append(chunk_text)
                break
            if "#" in chunk_text:
                output_text.append(chunk_text)
                continue
            else:
                sentence_splits = chunk_text.split("。")
                output_sent = []
                for sentence in sentence_splits:
                    query_result = embeddings.embed_query(sentence)
                    similarities = cosine_similarity([query_result], para_result).reshape((-1,))
                    # para_ids
                    sorted_ix = np.argsort(-similarities)
                    idx = sorted_ix[0]
                    source = paragraphs[idx]
                    # to skip white space
                    if len(sentence.strip()) > 0:
                        if not self.is_punctuation(sentence[-1]):
                            sentence += "。"
                        if similarities[idx] >= 0.9:
                            sentence += (
                                f"<sup>[\\[{url_index[source['url']]['index']}\\]]({source['url']})</sup>"
                            )
                    output_sent.append(sentence)
                chunk_text = "".join(output_sent)
                output_text.append(chunk_text)
        return "\n\n".join(output_text)
