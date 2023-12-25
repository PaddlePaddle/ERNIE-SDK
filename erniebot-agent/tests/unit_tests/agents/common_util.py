import pytest
from tests.unit_tests.testing_utils.mocks.mock_tool import FakeTool


@pytest.fixture(scope="module")
def identity_tool():
    return FakeTool(
        name="identity_tool",
        description="This tool simply forwards the input.",
        parameters={
            "type": "object",
            "properties": {
                "param": {
                    "type": "string",
                    "description": "Input parameter.",
                }
            },
        },
        responses={
            "type": "object",
            "properties": {
                "param": {
                    "type": "string",
                    "description": "Same as the input parameter.",
                }
            },
        },
        function=lambda param: {"param": param},
    )


@pytest.fixture(scope="module")
def no_input_no_output_tool():
    return FakeTool(
        name="no_input_no_output_tool",
        description="This tool takes no input parameters and returns no output parameters.",
        parameters={"type": "object", "properties": {}},
        responses={"type": "object", "properties": {}},
        function=lambda: {},
    )


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


NO_EXAMPLE_RESPONSE = {
    "logId": "2dc5d9018f912bb4c62f2653bdf05424",
    "errorCode": 0,
    "errorMsg": "Success",
    "timestamp": 1703208782306,
    "result": [],
}
