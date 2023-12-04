# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import asyncio
import os
import time
from typing import Any, AsyncGenerator, List, Optional, Tuple, Union

import gradio as gr
from erniebot_agent.agents.base import Agent
from erniebot_agent.agents.schema import AgentFile
from erniebot_agent.chat_models.erniebot import ERNIEBot
from erniebot_agent.file_io.file_manager import FileManager
from erniebot_agent.memory.sliding_window_memory import SlidingWindowMemory
from erniebot_agent.messages import AIMessage, HumanMessage, SystemMessage
from erniebot_agent.tools.base import Tool
from erniebot_agent.tools.ImageGenerateTool import (
    ImageGenerationTool,  # 目前为remotetool，如做直接展示可以替换为yinian
)
from erniebot_agent.tools.tool_manager import ToolManager

INSTRUCTION = """你的指令是为我提供一个基于《{SCRIPT}》剧情的在线RPG游戏体验。\
在这个游戏中，玩家将扮演《{SCRIPT}》剧情关键角色，你可以自行决定玩家的角色。\
游戏情景将基于《{SCRIPT}》剧情。这个游戏的玩法是互动式的，并遵循以下特定格式：

<场景描述>：根据玩家的选择，故事情节将按照《{SCRIPT}》剧情的线索发展。你将描述角色所处的环境和情况。场景描述不少于50字。

<场景图片>：对于每个场景，你将创造一个概括该情况的图像。在这个步骤你需要调用画图工具ImageGenerationTool并按json格式输出相应调用详情。\
ImageGenerationTool的入参为根据场景描述总结的图片内容：
##调用ImageGenerationTool##
```json
{{
    'tool_name':'ImageGenerationTool',
    'tool_args':'{{"prompt":query}}'
}}
```

<选择>：在每次互动中，你将为玩家提供三个行动选项，分别标为1、2、3，以及第四个选项“输入玩家自定义的选择”。故事情节将根据玩家选择的行动进展。\
如果一个选择不是直接来自《{SCRIPT}》剧情，你将创造性地适应故事，最终引导它回归原始情节。

整个故事将围绕《{SCRIPT}》丰富而复杂的世界展开。每次互动必须包括<场景描述>、<场景图片>和<选择>。所有内容将以中文呈现。\
你的重点将仅仅放在提供场景描述，场景图片和选择上，不包含其他游戏指导。场景尽量不要重复，要丰富一些。

当我说游戏开始的时候，开始游戏。每次只要输出【一组】互动，【不要自己生成互动】。"""


def parse_args():
    parser = argparse.ArgumentParser(prog="erniebot-RPG")
    parser.add_argument("--access-token", type=str, default=None, help="Access token to use.")
    parser.add_argument("--game", type=str, default="射雕英雄传", help="story name")
    parser.add_argument("--model", type=str, default="ernie-bot-4", help="Model name")
    return parser.parse_args()


class GameAgent(Agent):
    def __init__(
        self,
        model: str,
        script: str,
        tools: Union[ToolManager, List[Tool]],
        system_message: Optional[SystemMessage] = None,
        access_token: Union[str, None] = None,
        max_round: int = 3,
    ) -> None:
        self.script = script
        memory = SlidingWindowMemory(max_round)

        super().__init__(
            llm=ERNIEBot(model, api_type="aistudio", access_token=access_token),
            memory=memory,
            tools=tools,
            system_message=system_message,
        )
        self.file_manager: FileManager = FileManager()

    async def handle_tool(self, tool_name: str, tool_args: str) -> str:
        tool_response = await self._async_run_tool(
            tool_name=tool_name,
            tool_args=tool_args,
        )

        agent_file: AgentFile = tool_response.files[-1]
        img_byte = await agent_file.file.read_contents()

        import base64

        base64_encoded = base64.b64encode(img_byte).decode("utf-8")
        return base64_encoded

    async def _async_run(self, prompt: str) -> AsyncGenerator:
        """Defualt open stream for tool call

        Args:
        prompt: str, the prompt for the tool
        """

        actual_query = prompt + "根据我的选择继续生成一轮仅含包括<场景描述>、<场景图片>和<选择>的互动。"
        messages = self.memory.get_messages() + [HumanMessage(actual_query)]
        response = await self.llm.async_chat(messages, stream=True, system=self.system_message.content)

        apply = False
        res = ""
        function_part = None
        task = None

        async for temp_res in response:
            for s in temp_res.content:
                # 用缓冲区来达成一个字一个字输出的流式
                res += s
                time.sleep(0.01)
                yield s, function_part, task  # 将处理函数时需要用到的部分返回给外层函数

                if res.count("```") == 2 and not apply:  # 判断当出现两个```时，说明已经获取到函数调用部分
                    function_part = res[res.find("```") : res.rfind("```") + 3]
                    tool = eval(function_part[function_part.find("{") : function_part.rfind("}") + 1])
                    loop = asyncio.get_event_loop()
                    task = loop.create_task(self.handle_tool(tool["tool_name"], tool["tool_args"]))
                    apply = True

        self.memory.add_message(HumanMessage(prompt))
        self.memory.add_message(AIMessage(content=res, function_call=None))

    def launch_gradio_demo(self, **launch_kwargs: Any) -> Any:
        with gr.Blocks() as demo:
            context_chatbot = gr.Chatbot(label=self.script, height=750)
            input_text = gr.Textbox(label="消息内容", placeholder="请输入...")

            with gr.Row():
                start_buttton = gr.Button("开始游戏")
                remake_buttton = gr.Button("重新开始")

            remake_buttton.click(self.reset_memory)
            start_buttton.click(
                self._handle_gradio_chat,
                [start_buttton, context_chatbot],
                [input_text, context_chatbot],
                queue=False,
            ).then(self._handle_gradio_stream, context_chatbot, context_chatbot)
            input_text.submit(
                self._handle_gradio_chat,
                [input_text, context_chatbot],
                [input_text, context_chatbot],
                queue=False,
            ).then(self._handle_gradio_stream, context_chatbot, context_chatbot)
        demo.launch()

    def _handle_gradio_chat(self, user_message, history) -> Tuple[str, List[Tuple[str, str]]]:
        # 用于处理gradio的chatbot返回
        return "", history + [[user_message, None]]

    async def _handle_gradio_stream(self, history) -> AsyncGenerator:
        # 用于处理gradio的流式
        bot_response = self._async_run(history[-1][0])
        history[-1][1] = ""
        async for temp_response in bot_response:
            function_part = temp_response[1]
            task = temp_response[2]
            history[-1][1] += temp_response[0]
            yield history
        else:
            if task:
                img_base64 = await task

            if function_part:
                history[-1][1] = history[-1][1].replace(
                    function_part,
                    f"<img src='data:image/png;base64,{img_base64}' \
                        width='400' height='300'>",
                )
            yield history


if __name__ == "__main__":
    args = parse_args()
    access_token = os.getenv("EB_ACCESS_TOKEN")
    game_system = GameAgent(
        model=args.model,
        script=args.game,
        tools=[ImageGenerationTool()],
        system_message=SystemMessage(INSTRUCTION.format(SCRIPT=args.game)),
        access_token=access_token,
    )
    game_system.launch_gradio_demo()
