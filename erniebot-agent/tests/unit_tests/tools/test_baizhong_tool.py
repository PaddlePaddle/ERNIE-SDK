import unittest
from unittest.mock import patch

from erniebot_agent.retrieval.baizhong_search import BaizhongSearch
from erniebot_agent.tools.baizhong_tool import BaizhongSearchTool

FUNCTIONCALL_PARAMETERS = {
    "name": "BaizhongSearchTool",
    "description": "This is the search tool description",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "查询语句"},
            "top_k": {"type": "integer", "description": "返回结果数量"},
        },
        "required": ["query", "top_k"],
    },
    "responses": {
        "type": "object",
        "properties": {
            "documents": {
                "type": "array",
                "description": "检索结果，内容和用户输入query相关的段落",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "检索结果的文本的id"},
                        "title": {"type": "string", "description": "检索结果的标题"},
                        "document": {"type": "string", "description": "检索结果的内容"},
                    },
                    "required": ["id", "title", "document"],
                },
            }
        },
        "required": ["documents"],
    },
}


class TestBaizhongSearchTool(unittest.TestCase):
    @patch("requests.request")
    def setUp(self, mock_request):
        knowledge_base_name = "test"
        access_token = "your access token"
        knowledge_base_id = 111
        self.baizhong_db = BaizhongSearch(
            knowledge_base_name=knowledge_base_name,
            access_token=access_token,
            knowledge_base_id=knowledge_base_id if knowledge_base_id != "" else None,
        )

    def test_schema(self):
        tool_description = "This is the search tool description"
        calculator = BaizhongSearchTool(description=tool_description, db=self.baizhong_db)
        function_call_schema = calculator.function_call_schema()
        self.assertEqual(
            function_call_schema,
            FUNCTIONCALL_PARAMETERS,
        )
