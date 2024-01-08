import asyncio
from typing import Any, Dict, List, Type

from pydantic import Field

from erniebot_agent.agents import FunctionAgent
from erniebot_agent.agents.callback.default import get_no_ellipsis_callback
from erniebot_agent.chat_models.erniebot import ERNIEBot
from erniebot_agent.file import GlobalFileManagerHandler
from erniebot_agent.memory import AIMessage, HumanMessage, Message
from erniebot_agent.memory.sliding_window_memory import SlidingWindowMemory
from erniebot_agent.tools.base import Tool
from erniebot_agent.tools.schema import ToolParameterView


class TextRepeaterToolInputView(ToolParameterView):
    input_file_id: str = Field(description="输入的文件ID")
    repeat_times: int = Field(description="重复次数")


class TextRepeaterToolOutputView(ToolParameterView):
    text: str = Field(description="重复后的文本")


class TextRepeaterTool(Tool):
    description: str = "TextRepeaterTool用于将输入文件中的前10个字进行指定次数的重复并输出。"
    input_type: Type[ToolParameterView] = TextRepeaterToolInputView
    ouptut_type: Type[ToolParameterView] = TextRepeaterToolOutputView

    async def __call__(self, input_file_id: str, repeat_times: int) -> Dict[str, Any]:
        if "<split>" in input_file_id:
            input_file_id = input_file_id.split("<split>")[0]

        file_manager = await GlobalFileManagerHandler().get()
        input_file = file_manager.look_up_file_by_id(input_file_id)
        if input_file is None:
            raise RuntimeError("File not found")
        # text = (await input_file.read_contents())[:10]
        text = "测试"
        text *= repeat_times
        return {"result": text}

    @property
    def examples(self) -> List[Message]:
        return [
            HumanMessage(
                "请把文件中的前10个字复制三遍返回。这句话中包含的文件如下：\
\n<file>file-local-609f02c0-98c3-11ee-a72b-fa2020087eb4<split>浅谈牛奶的营养与消费趋势.docx\
</file><url>https://qianfan-doc.bj.bcebos.com/chatfile/%E6%B5%85%E8%B0%88%E7%89%9B%E5%A5\
%B6%E7%9A%84%E8%90%A5%E5%85%BB%E4%B8%8E%E6%B6%88%E8%B4%B9%E8%B6%8B%E5%8A%BF.docx</url>"
            ),
            AIMessage(
                "",
                function_call={
                    "name": self.tool_name,
                    "thoughts": f"用户想知道文件中的前10个字重复3次是什么，\
我可以使用{self.tool_name}工具来获得重复结果，其中`input_file_id`字段的内容为：\
'file-local-609f02c0-98c3-11ee-a72b-fa2020087eb4<split>浅谈牛奶的营养与消费趋势.docx'，`repeat_times`字段的内容为3。",
                    "arguments": '{"input_file_id": \
"file-local-609f02c0-98c3-11ee-a72b-fa2020087eb4<split>浅谈牛奶的营养与消费趋势.docx", "repeat_times": 3}',
                },
                token_usage={
                    "prompt_tokens": 5,
                    "completion_tokens": 7,
                },
            ),
        ]


class TextRepeaterNoFileToolInputView(ToolParameterView):
    text: str = Field(description="输入的文本")
    repeat_times: int = Field(description="重复次数")


class TextRepeaterNoFileToolOutputView(ToolParameterView):
    text: str = Field(description="重复后的文本")


class TextRepeaterNoFileTool(Tool):
    description: str = "TextRepeaterNoFileTool用于将输入文本进行指定次数的重复并输出。"
    input_type: Type[ToolParameterView] = TextRepeaterNoFileToolInputView
    ouptut_type: Type[ToolParameterView] = TextRepeaterNoFileToolOutputView

    async def __call__(self, text, repeat_times: int) -> Dict[str, Any]:
        text *= repeat_times
        return {"result": text}

    @property
    def examples(self) -> List[Message]:
        return [
            HumanMessage("请把文本“三十多发发”复制三遍返回。"),
            AIMessage(
                "",
                function_call={
                    "name": self.tool_name,
                    "thoughts": f"用户想知道文本“三十多发发”重复3次是什么，\
我可以使用{self.tool_name}工具来获得重复结果，其中`text`字段的内容为：'三十多发发'，`repeat_times`字段的内容为3。",
                    "arguments": '{"text": "三十多发发", "repeat_times": 3}',
                },
                token_usage={
                    "prompt_tokens": 5,
                    "completion_tokens": 7,
                },
            ),
        ]


class get_current_weatherInputView(ToolParameterView):
    location: str = Field(description="省，市名，例如：河北省，石家庄")
    unit: str = Field(description="重复次数")


class get_current_weather(Tool):
    description: str = "获得指定地点的天气"
    input_type: Type[ToolParameterView] = get_current_weatherInputView

    async def __call__(self, location, unit: int = None) -> None:
        return None


# TODO(shiyutang): replace this when model is online
llm = ERNIEBot(model="ernie-3.5", api_type="custom", enable_multi_step_tool_call=True)
memory = SlidingWindowMemory(max_round=1)
plugins = ["ChatFile", "eChart"]
# plugins: List[str] = []
agent = FunctionAgent(
    llm=llm,
    tools=[
        get_current_weather(),
    ],
    memory=memory,
    callbacks=get_no_ellipsis_callback(),
    plugins=plugins,
)


async def run_agent():
    await GlobalFileManagerHandler().configure(
        enable_remote_file=True,
        access_token="your-access-token",
    )
    file_manager = await GlobalFileManagerHandler().get()

    docx_file = await file_manager.create_file_from_path(
        file_path="浅谈牛奶的营养与消费趋势.docx",
        file_type="remote",
    )

    print("=" * 10 + "echart返回结果" + "=" * 10 + "\n")  # ok
    agent_resp = await agent.run("帮我画一个饼状图：8月的用户反馈中，BUG有100条，需求有100条，使用咨询100条，总共300条反馈")  # ok
    print(agent_resp.text)
    print("\n" + "=" * 20 + "\n")

    print("=" * 10 + "喝牛奶的好处" + "=" * 10 + "\n")
    agent_resp = await agent.run("喝牛奶有什么好处", files=[docx_file])  # ok
    print(agent_resp.text)
    print("\n" + "=" * 20 + "\n")

    print("=" * 10 + "传入plugins，docx不使用chatFile、使用Tools的返回结果" + "=" * 10 + "\n")  # ok
    agent_resp = await agent.run("请把文件中的前10个字复制三遍返回", files=[docx_file])
    print(agent_resp.text)
    print("\n" + "=" * 20 + "\n")

    print("=" * 10 + "混合编排" + "=" * 10 + "\n")
    agent_resp = await agent.run("请把文件中的前10个字复制三遍，并将结果和文档一起创作一篇短文", files=[docx_file])  # 没有联排
    print(agent_resp.text)
    print(agent_resp.annotations)
    print("\n" + "=" * 20 + "\n")

    print("=" * 10 + "echart不带File混合编排" + "=" * 10 + "\n")  # ok
    agent_resp = await agent.run(
        '请告诉我"今天是美好的一天"重复三遍是什么？然后画一个饼状图：8月的用户反馈中，BUG有100条，需求有100条，使用咨询100条，总共300条反馈'
    )
    print(agent_resp.text)
    print("\n" + "=" * 20 + "\n")

    # TODO：多轮调用


if __name__ == "__main__":
    asyncio.run(run_agent())
