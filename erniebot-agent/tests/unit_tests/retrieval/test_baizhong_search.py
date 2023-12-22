import unittest
from unittest.mock import MagicMock, patch

from erniebot_agent.retrieval.baizhong_search import BaizhongSearch

EXAMPLE_RESPONSE = {
    "logId": "2dc5d9018f912bb4c62f2653bdf05424",
    "errorCode": 0,
    "errorMsg": "Success",
    "timestamp": 1703208782306,
    "result": [
        {
            "score": 0.01162862777709961,
            "fileId": "495735246643269",
            "source": {
                "doc": '{"_doc_id":"495735246643269","_id":"495735246643269", \
                    "content_se":"住房和城乡建设部规章城市管理执法办法", \
                    "knowledgeBaseId":495735236530245,"title":"城市管理执法办法.pdf"}',
                "es_score": 0.5613938,
                "para": "住房和城乡建设部规章城市管理执法办法",
                "title": "城市管理执法办法.pdf",
            },
        },
        {
            "score": 0.011362016201019287,
            "fileId": "495735246643270",
            "source": {
                "doc": '{"_doc_id":"495735246643270","_id":"495735246643270", \
                "content_se":"城市管理执法主管部门应当定期开展执法人员的培训和考核。", \
                "knowledgeBaseId":495735236530245,"title":"城市管理执法办法.pdf"}',
                "es_score": 0.5550896,
                "para": "城市管理执法主管部门应当定期开展执法人员的培训和考核。",
                "title": "城市管理执法办法.pdf",
            },
        },
    ],
}


SEARCH_RESULTS = [
    {
        "id": "495735246643269",
        "content": "住房和城乡建设部规章城市管理执法办法",
        "title": "城市管理执法办法.pdf",
        "score": 0.01162862777709961,
    },
    {
        "id": "495735246643270",
        "content": "城市管理执法主管部门应当定期开展执法人员的培训和考核。",
        "title": "城市管理执法办法.pdf",
        "score": 0.011362016201019287,
    },
]

KNOWLEDGEBASE_RESPONESE = {
    "logId": "2dc5d9018f912bb4c62f2653bdf05424",
    "errorCode": 0,
    "errorMsg": "Success",
    "timestamp": 1703208782306,
    "result": {"knowledgeBaseId": "123456", "knowledgeBaseName": "test"},
}


class TestBaizhongSearch(unittest.TestCase):
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

    @patch("requests.post")
    def test_create_knowledge_base(self, mock_request):
        knowledge_base_name = "test"
        mock_response = MagicMock(status_code=200, json=lambda: KNOWLEDGEBASE_RESPONESE)
        mock_request.return_value = mock_response
        access_token = "your access token"
        knowledge_base_id = ""
        baizhong_db = BaizhongSearch(
            knowledge_base_name=knowledge_base_name,
            access_token=access_token,
            knowledge_base_id=knowledge_base_id if knowledge_base_id != "" else None,
        )

        self.assertEqual(baizhong_db.knowledge_base_id, "123456")

    @patch("requests.post")
    def test_search(self, mock_request):
        mock_response = MagicMock(status_code=200, json=lambda: EXAMPLE_RESPONSE)
        mock_request.return_value = mock_response
        response = self.baizhong_db.search(query="Hello")
        self.assertEqual(len(response), 2)
        self.assertEqual(
            response,
            SEARCH_RESULTS,
        )
