import asyncio
from typing import Any, Dict, Type

from erniebot_agent.agents.callback.default import get_no_ellipsis_callback
from erniebot_agent.agents.functional_agent import FunctionalAgent
from erniebot_agent.chat_models.erniebot import ERNIEBot
from erniebot_agent.file_io import get_file_manager
from erniebot_agent.memory.whole_memory import WholeMemory
from erniebot_agent.tools.base import Tool
from erniebot_agent.tools.schema import ToolParameterView
from pydantic import Field


class TextRepeaterToolInputView(ToolParameterView):
    input_file_id: str = Field(description="输入文件ID")
    repeat_times: int = Field(description="重复次数")


class TextRepeaterToolOutputView(ToolParameterView):
    text: str = Field(description="重复后的文本")


class TextRepeaterTool(Tool):
    description: str = "TextRepeaterTool用于将输入文件中的前10个字进行指定次数的重复并输出。"
    input_type: Type[ToolParameterView] = TextRepeaterToolInputView
    ouptut_type: Type[ToolParameterView] = TextRepeaterToolOutputView

    async def __call__(self, input_file_id: str, repeat_times: int) -> Dict[str, Any]:
        file_manager = get_file_manager()
        input_file = file_manager.look_up_file_by_id(input_file_id)
        if input_file is None:
            raise RuntimeError("File not found")
        text = (await input_file.read_contents()).decode("utf-8")[:10]
        text *= repeat_times
        return {"result": text}


llm = ERNIEBot(model="ernie-bot", api_type="custom")
# export EB_BASE_URL="http://None/ernie-foundry/v1" # /erniebot/plugins_v3
# export EB_BASE_URL="http://None/ernie-foundry/v1"
memory = WholeMemory()
file_manager = get_file_manager()
agent = FunctionalAgent(
    llm=llm,
    tools=[TextRepeaterTool()],
    memory=memory,
    file_manager=file_manager,
    callbacks=get_no_ellipsis_callback(),
)


async def run_agent():
    docx_file = await file_manager.create_file_from_path(
        file_path="浅谈牛奶的营养与消费趋势.docx",
        file_type="local",
        URL="https://qianfan-doc.bj.bcebos.com/chatfile/ \
            %E6%B5%85%E8%B0%88%E7%89%9B%E5%A5%B6%E7%9A%84%E\
            8%90%A5%E5%85%BB%E4%B8%8E%E6%B6%88%E8%B4%B9%E8%B6%8B%E5%8A%BF.docx",
    )
    # test echart
    print("=" * 10 + "echart返回结果" + "=" * 10 + "\n")
    agent_resp = await agent.async_run("帮我画一个饼状图：8月的用户反馈中，BUG有100条，需求有100条，使用咨询100条，总共300条反馈")  # ok
    print(agent_resp.text)
    print("\n" + "=" * 20 + "\n")
    # test use chatfile with 牛奶.docx
    print("=" * 10 + "喝牛奶的好处" + "=" * 10 + "\n")
    # import pdb;pdb.set_trace()
    agent_resp = await agent.async_run("喝牛奶有什么好处", files=[docx_file])  # ok
    print(agent_resp.text)
    print("\n" + "=" * 20 + "\n")
    # test not use chatfile with 牛奶.docx
    print("=" * 10 + "docx不使用chatFile的返回结果" + "=" * 10 + "\n")
    agent_resp = await agent.async_run("请把文件中的前10个字复制三遍返回", files=[docx_file])  # 没有调用tool，依旧是摘要
    print(agent_resp.text)
    print("\n" + "=" * 20 + "\n")
    # test use 混合编排
    print("=" * 10 + "混合编排" + "=" * 10 + "\n")
    agent_resp = await agent.async_run("请把文件中的前10个字复制三遍，并将结果和文档一起创作一篇短文", files=[docx_file])  # 没有联排
    print(agent_resp.text)
    print(agent_resp.annotations)
    print("\n" + "=" * 20 + "\n")


if __name__ == "__main__":
    asyncio.run(run_agent())
    # agent.launch_gradio_demo(server_name="0.0.0.0", server_port=8017, share=True) # gradio 还没加上文件上传等
