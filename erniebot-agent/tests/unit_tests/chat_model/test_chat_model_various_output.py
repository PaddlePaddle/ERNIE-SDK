import pytest
from erniebot.response import EBResponse
from erniebot import ChatCompletionResponse

from erniebot_agent.chat_models.erniebot import convert_response_to_output
from erniebot_agent.memory import HumanMessage
from erniebot_agent.memory.messages import AIMessage
from tests.unit_tests.testing_utils.mocks.mock_chat_models import (
    FakeERNIEBotWithPresetResponses,
)

from erniebot_agent.memory import HumanMessage

# 1. fake various output from erniebot
@pytest.fixture
def fake_erniebot_with_search_info():
    rcode = (200,)
    rbody = {
        "id": "as-a4svw1crr0",
        "object": "chat.completion",
        "created": 1703492795,
        "result": "",
        "usage": {"prompt_tokens": 1285, "completion_tokens": 124, "total_tokens": 1409},
        "is_truncated": False,
        "finish_reason": "function_call",
        "need_clear_history": False,
        "search_info": {"search_results": [{"index": 1, "url": "", "title": "深圳天气预报_一周天气预报"}]},
    }
    rheaders = {
        "Connection": "keep-alive",
        "Content-Security-Policy": "frame-ancestors https://*.baidu.com/",
        "Content-Type": "application/json",
        "Date": "Mon, 25 Dec 2023 08:26:36 GMT",
        "Server": "nginx",
        "Statement": "AI-generated",
        "Vary": "Origin",
        "X-Frame-Options": "allow-from https://*.baidu.com/",
        "X-Request-Id": "6efe63c60902b1459106b21cb6de566a",
        "Transfer-Encoding": "chunked",
    }
    fake_response_with_search_info = ChatCompletionResponse(rcode, rbody, rheaders)
    return FakeERNIEBotWithPresetResponses(
        [convert_response_to_output(fake_response_with_search_info, AIMessage)]
    )


@pytest.fixture
def fake_erniebot_with_plugin_info():  # withfile plugins
    rcode = (200,)
    rbody = {
        "id": "as-4tpm80hx7c",
        "object": "chat.completion",
        "created": 1703496289,
        "result": '\n\n ```echarts-option\n[{"series":[{"type":"pie","name":"8月","data":[{"name":"BUG","value":100},{"name":"需求","value":100},{"name":"使用咨询","value":100}],"label":{"show":true,"formatter":"{b}：{c} 条"}}],"title":{"text":"8月的用户反馈（8月）"},"tooltip":{"show":true},"legend":{"show":true,"bottom":15}}]\n```\n\n\n\n**图表数据:**\n\n| 反馈类型 | 8月 |\n| :--: |  :--: |\n| BUG | 100 |\n| 需求 | 100 |\n| 使用咨询 | 100 |\n\n我（文心一言）是百度开发的人工智能模型，通过分析大量公开文本信息进行学习。然而，我所提供的信息可能存在误差。因此上文内容仅供参考，并不应被视为专业建议。',
        "is_truncated": False,
        "need_clear_history": False,
        "plugin_info": [
            {
                "plugin_id": "1004:1.0.3",
                "plugin_name": "",
                "plugin_req": '{"data":"BUG有100条，需求有100条，使用咨询100条，总共300条反馈","query":"帮我画一个饼状图：8月的用户反馈中，BUG有100条，需求有100条，使用咨询100条，总共300条反馈","chartType":"pie","title":"8月的用户反馈","last_bot_message":""}',
                "plugin_resp": "",
                "status": "1",
                "api_id": "eChart.plot",
            },
            {
                "plugin_id": "1004:1.0.3",
                "plugin_name": "",
                "plugin_req": '{"data":"BUG有100条，需求有100条，使用咨询100条，总共300条反馈","query":"帮我画一个饼状图：8月的用户反馈中，BUG有100条，需求有100条，使用咨询100条，总共300条反馈","chartType":"pie","title":"8月的用户反馈","last_bot_message":""}',
                "plugin_resp": '{"errCode":0,"errMsg":"success","status":"","actionName":"数据分析","actionContent":"数据分析中"}',
                "status": "1",
                "api_id": "eChart.plot",
            },
            {
                "plugin_id": "1004:1.0.3",
                "plugin_name": "",
                "plugin_req": '{"data":"BUG有100条，需求有100条，使用咨询100条，总共300条反馈","query":"帮我画一个饼状图：8月的用户反馈中，BUG有100条，需求有100条，使用咨询100条，总共300条反馈","chartType":"pie","title":"8月的用户反馈","last_bot_message":""}',
                "plugin_resp": '{"errCode":0,"errMsg":"success","status":"","actionName":"数据分析","actionContent":"数据分析中"}',
                "status": "1",
                "api_id": "eChart.plot",
            },
            {
                "plugin_id": "1004:1.0.3",
                "plugin_name": "",
                "plugin_req": '{"data":"BUG有100条，需求有100条，使用咨询100条，总共300条反馈","query":"帮我画一个饼状图：8月的用户反馈中，BUG有100条，需求有100条，使用咨询100条，总共300条反馈","chartType":"pie","title":"8月的用户反馈","last_bot_message":""}',
                "plugin_resp": '{"errCode":0,"errMsg":"success","status":"","actionName":"数据分析","actionContent":"数据分析中"}\n{"errCode":0,"errMsg":"success","status":"","actionName":"数据分析","actionContent":"数据分析成功"}',
                "status": "1",
                "api_id": "eChart.plot",
            },
            {
                "plugin_id": "1004:1.0.3",
                "plugin_name": "",
                "plugin_req": '{"data":"BUG有100条，需求有100条，使用咨询100条，总共300条反馈","query":"帮我画一个饼状图：8月的用户反馈中，BUG有100条，需求有100条，使用咨询100条，总共300条反馈","chartType":"pie","title":"8月的用户反馈","last_bot_message":""}',
                "plugin_resp": '{"errCode":0,"errMsg":"success","status":"","actionName":"数据分析","actionContent":"数据分析中"}\n{"errCode":0,"errMsg":"success","status":"","actionName":"数据分析","actionContent":"数据分析成功"}\n{"errCode":0,"errMsg":"success","status":"","actionName":"图表绘制","actionContent":"图表绘制中"}',
                "status": "1",
                "api_id": "eChart.plot",
            },
            {
                "plugin_id": "1004:1.0.3",
                "plugin_name": "",
                "plugin_req": '{"data":"BUG有100条，需求有100条，使用咨询100条，总共300条反馈","query":"帮我画一个饼状图：8月的用户反馈中，BUG有100条，需求有100条，使用咨询100条，总共300条反馈","chartType":"pie","title":"8月的用户反馈","last_bot_message":""}',
                "plugin_resp": '{"errCode":0,"errMsg":"success","status":"","actionName":"数据分析","actionContent":"数据分析中"}\n{"errCode":0,"errMsg":"success","status":"","actionName":"数据分析","actionContent":"数据分析成功"}\n{"errCode":0,"errMsg":"success","status":"","actionName":"图表绘制","actionContent":"图表绘制中"}\n{"errCode":0,"errMsg":"success","status":"","actionName":"图表绘制","actionContent":"图表绘制成功"}',
                "status": "1",
                "api_id": "eChart.plot",
            },
            {
                "plugin_id": "1004:1.0.3",
                "plugin_name": "",
                "plugin_req": '{"data":"BUG有100条，需求有100条，使用咨询100条，总共300条反馈","query":"帮我画一个饼状图：8月的用户反馈中，BUG有100条，需求有100条，使用咨询100条，总共300条反馈","chartType":"pie","title":"8月的用户反馈","last_bot_message":""}',
                "plugin_resp": '{"errCode":0,"errMsg":"success","status":"","actionName":"数据分析","actionContent":"数据分析中"}\n{"errCode":0,"errMsg":"success","status":"","actionName":"数据分析","actionContent":"数据分析成功"}\n{"errCode":0,"errMsg":"success","status":"","actionName":"图表绘制","actionContent":"图表绘制中"}\n{"errCode":0,"errMsg":"success","status":"","actionName":"图表绘制","actionContent":"图表绘制成功"}\n{"errCode":0,"option":"REMOVED","usage":null}\n{"errCode":0,"errMsg":"success","status":"","actionName":"图表解释","actionContent":"图表解释成功"}',
                "status": "1",
                "api_id": "eChart.plot",
            },
            {
                "plugin_id": "1004:1.0.3",
                "plugin_name": "",
                "plugin_req": '{"data":"BUG有100条，需求有100条，使用咨询100条，总共300条反馈","query":"帮我画一个饼状图：8月的用户反馈中，BUG有100条，需求有100条，使用咨询100条，总共300条反馈","chartType":"pie","title":"8月的用户反馈","last_bot_message":""}',
                "plugin_resp": '{"errCode":0,"errMsg":"success","status":"","actionName":"数据分析","actionContent":"数据分析中"}\n{"errCode":0,"errMsg":"success","status":"","actionName":"数据分析","actionContent":"数据分析成功"}\n{"errCode":0,"errMsg":"success","status":"","actionName":"图表绘制","actionContent":"图表绘制中"}\n{"errCode":0,"errMsg":"success","status":"","actionName":"图表绘制","actionContent":"图表绘制成功"}\n{"errCode":0,"option":"REMOVED","usage":null}\n{"errCode":0,"errMsg":"success","status":"","actionName":"图表解释","actionContent":"图表解释成功"}\n{"errCode":0,"option":"REMOVED","usage":{"len_answer":0,"len_chart_interpret":161,"len_echarts_dict_str":224,"total_tokens":423}}',
                "status": "2",
                "api_id": "eChart.plot",
            },
        ],
        "plugin_metas": [
            {
                "apiId": "eChart.plot",
                "logoUrl": "https://echarts.bj.bcebos.com/echarts-logo.png",
                "operationId": "plot",
                "pluginId": "1004:1.0.3",
                "pluginNameForHuman": "E言易图",
                "pluginNameForModel": "eChart",
                "pluginVersion": "1.0.3",
                "runtimeMetaInfo": {
                    "function_call": {
                        "arguments": '{"query":"帮我画一个饼状图：8月的用户反馈中，BUG有100条，需求有100条，使用咨询100条，总共300条反馈","title":"8月的用户反馈","chartType":"pie","data":"BUG有100条，需求有100条，使用咨询100条，总共300条反馈"}',
                        "name": "eChart.plot",
                        "thoughts": "这是一个图表绘制需求",
                    },
                    "returnRawFieldName": "option",
                    "thoughts": "这是一个图表绘制需求",
                },
                "uiMeta": None,
            }
        ],
        "usage": {
            "prompt_tokens": 875,
            "completion_tokens": 172,
            "total_tokens": 1047,
            "plugins": [
                {
                    "name": "eChart",
                    "parse_tokens": 0,
                    "abstract_tokens": 0,
                    "search_tokens": 0,
                    "total_tokens": 423,
                }
            ],
        },
    }
    rheaders = {
        "Content-Type": "application/json; charset=utf-8",
        "Statement": "AI-generated",
        "X-Aipe-Self-Def": "eb_total_tokens:1047,prompt_tokens:875, echart_total_tokens:423,id:as-4tpm80hx7c",
        "Date": "Mon, 25 Dec 2023 09:24:49 GMT",
        "Transfer-Encoding": "chunked",
    }
    fake_response_with_plugin_info = ChatCompletionResponse(rcode, rbody, rheaders)

    return FakeERNIEBotWithPresetResponses(
        [convert_response_to_output(fake_response_with_plugin_info, AIMessage)]
    )


@pytest.fixture
def fake_erniebot_with_function_call():
    rcode = 200
    rbody = {
        "id": "as-a4svw1crr0",
        "object": "chat.completion",
        "created": 1703492795,
        "result": "",
        "usage": {"prompt_tokens": 1285, "completion_tokens": 124, "total_tokens": 1409},
        "is_truncated": False,
        "finish_reason": "function_call",
        "need_clear_history": False,
        "function_call": {
            "name": "TextRepeaterNoFileTool",
            "thoughts": '用户想要知道"今天是美好的一天"重复三遍是什么，然后生成一个饼状图。首先，我需要使用TextRepeaterNoFileTool工具来获取重复后的文本，然后使用CalculatorTool工具来计算饼状图的数值。任务拆解：[sub-task1: 使用TextRepeaterNoFileTool工具重复文本，sub-task2: 使用CalculatorTool工具计算饼状图]。接下来，我将调用[TextRepeaterNoFileTool]来获取重复后的文本。',
            "arguments": '{"text":"今天是美好的一天","repeat_times":3}',
        },
    }
    rheaders = {
        "Connection": "keep-alive",
        "Content-Security-Policy": "frame-ancestors https://*.baidu.com/",
        "Content-Type": "application/json",
        "Date": "Mon, 25 Dec 2023 08:26:36 GMT",
        "Server": "nginx",
        "Statement": "AI-generated",
        "Vary": "Origin",
        "X-Frame-Options": "allow-from https://*.baidu.com/",
        "X-Request-Id": "6efe63c60902b1459106b21cb6de566a",
        "Transfer-Encoding": "chunked",
    }
    fake_function_call_response = ChatCompletionResponse(rcode, rbody, rheaders)
    return FakeERNIEBotWithPresetResponses(
        [convert_response_to_output(fake_function_call_response, AIMessage)]
    )


@pytest.fixture
def fake_erniebot_with_vanilla_output():
    rcode = 200
    rbody = {
        "id": "as-ieb0xfdmsv",
        "object": "chat.completion",
        "created": 1703493296,
        "result": "您好，我是文心一言，英文名是ERNIE Bot。我能够与人对话互动，回答问题，协助创作，高效便捷地帮助人们获取信息、知识和灵感。",
        "usage": {"prompt_tokens": 1236, "completion_tokens": 37, "total_tokens": 1273},
        "is_truncated": False,
        "finish_reason": "normal",
        "need_clear_history": False,
    }
    rheaders = {
        "Connection": "keep-alive",
        "Content-Security-Policy": "frame-ancestors https://*.baidu.com/",
        "Content-Type": "application/json",
        "Date": "Mon, 25 Dec 2023 08:34:56 GMT",
        "Server": "nginx",
        "Statement": "AI-generated",
        "Vary": "Origin",
        "X-Frame-Options": "allow-from https://*.baidu.com/",
        "X-Request-Id": "6bd541acb2c64765c5094df7a8485b7a",
        "Transfer-Encoding": "chunked",
    }
    fake_vanilla_message_response = ChatCompletionResponse(rcode, rbody, rheaders)
    return FakeERNIEBotWithPresetResponses(
        [convert_response_to_output(fake_vanilla_message_response, AIMessage)]
    )


# 2. tests each output independently
@pytest.mark.asyncio
async def test_erniebot_with_search_info(fake_erniebot_with_search_info):
    fake_erniebot = fake_erniebot_with_search_info
    messages = [HumanMessage("今天深圳天气怎么样？")]
    response = await fake_erniebot.chat(
        messages,
    )

    assert len(response.search_info) > 0


@pytest.mark.asyncio
async def test_erniebot_with_plugin_info(fake_erniebot_with_plugin_info):
    fake_erniebot = fake_erniebot_with_plugin_info
    messages = [HumanMessage("今天深圳天气怎么样？")]
    response = await fake_erniebot.chat(
        messages,
    )

    assert len(response.plugin_info) > 0


@pytest.mark.asyncio
async def test_erniebot_with_function_call(fake_erniebot_with_function_call):
    fake_erniebot = fake_erniebot_with_function_call
    messages = [HumanMessage("今天深圳天气怎么样？")]
    response = await fake_erniebot.chat(
        messages,
    )

    assert len(response.function_call) > 0


@pytest.mark.asyncio
async def test_erniebot_with_vanilla_output(fake_erniebot_with_vanilla_output):
    fake_erniebot = fake_erniebot_with_vanilla_output
    messages = [HumanMessage("今天深圳天气怎么样？")]
    response = await fake_erniebot.chat(
        messages,
    )

    assert response.plugin_info is None
    assert response.function_call is None
    assert response.search_info is None
