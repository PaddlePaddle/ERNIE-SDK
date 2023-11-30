#!/usr/bin/env python

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
import queue
import threading
import time
from typing import Any, AsyncGenerator, List, Optional, Tuple, Union

import gradio as gr
from erniebot_agent.agents.base import Agent
from erniebot_agent.chat_models.erniebot import ERNIEBot
from erniebot_agent.memory.base import Memory
from erniebot_agent.messages import AIMessage, HumanMessage, Message, SystemMessage
from erniebot_agent.tools.base import Tool
from erniebot_agent.tools.ImageGenerateTool import ImageGenerationTool
from erniebot_agent.tools.tool_manager import ToolManager
from erniebot_agent.utils.logging import logger

import erniebot as eb

INSTRUCTION = """你的指令是为我提供一个基于《{SCRIPT}》剧情的在线RPG游戏体验。\
在这个游戏中，玩家将扮演《{SCRIPT}》剧情关键角色，你可以自行决定玩家的角色。\
游戏情景将基于《{SCRIPT}》剧情。这个游戏的玩法是互动式的，并遵循以下特定格式：

<场景描述>：根据玩家的选择，故事情节将按照《{SCRIPT}》剧情的线索发展。你将描述角色所处的环境和情况。剧情发展请尽量快，场景描述不少于30字。

<场景图片>：对于每个场景，你将创造一个概括该情况的图像。在这个步骤你需要调用画图工具ImageGenerationTool。\
ImageGenerationTool的入参为根据场景描述总结的图片内容，请按json的格式输出：
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

SYSTEM_MESSAGE = "你是《{SCRIPT}》沉浸式图文RPG场景助手，能够生成图文剧情。\
                并给出玩家选项，整个故事将围绕《{SCRIPT}》丰富而复杂的世界展开。\
                每次互动必须包括<场景描述>、<场景图片>(需调用ImageGenerationTool并填写json)和<选择>。\
                每次仅生成一轮互动，不要自己生成玩家的选择"

# 创建消息队列用于传递文件地址
FILE_QUEUE: queue.Queue[ToolResponse] = queue.Queue()


def parse_args():
    parser = argparse.ArgumentParser(prog="erniebot-RPG")
    parser.add_argument("--access-token", type=str, default=None, help="Access token to use.")
    parser.add_argument("--game", type=str, default="射雕英雄传", help="story name")
    parser.add_argument("--model", type=str, default="ernie-bot-4", help="Model name")
    parser.add_argument(
        "--db-dir",
        type=str,
        default="/Users/tanzhehao/Documents/ERINE/ERNIE-Bot-SDK/examples/douluo_index_hf",
    )
    return parser.parse_args()


class SlidingWindowMemory(Memory):
    """This class controls max number of messages."""

    def __init__(self, max_num_message: int):
        super().__init__()
        self.max_num_message = max_num_message

        assert (isinstance(max_num_message, int)) and (
            max_num_message > 0
        ), "max_num_message should be positive integer, but got {max_token_limit}".format(
            max_token_limit=max_num_message
        )

    def add_message(self, message: Message):
        super().add_message(message=message)
        self.prune_message()

    def prune_message(self):
        # 保留第一轮的对话用于指令
        while len(self.get_messages()) > (self.max_num_message + 1) * 2:
            # 需修改memory的pop_message方法，支持将消息从内存中按索引删除
            self.msg_manager.pop_message(2)
            # `messages` must have an odd number of elements.
            if len(self.get_messages()) % 2 == 0:
                self.msg_manager.pop_message(2)


# def run_tool(tool) -> None:
#     # TODO 原有的tool调用方法，暂时没用到，但现有的tool调用方法无法直接完成
#     try:
#         generatetool = ImageGenerationTool()  # 实例化的tool需要和prompt对应
#         generatetool # mypy报错
#         img_byte = asyncio.run(eval(tool))
#         all_files = os.listdir("/private/var/folders/gw/lbw__qt16dl3sdh_5cgv_jl00000gn/T/gradio/")

#         # 用bytes会导致页面卡死，暂时还是使用location的方式
#         num_png_files = 0
#         for file in all_files:
#             if file.endswith(".png"):
#                 num_png_files += 1
#         save_path = (
#             f"/private/var/folders/gw/lbw__qt16dl3sdh_5cgv_jl00000gn/T/gradio/temp_{num_png_files+1}.png"
#         )
#         bytestr_to_png(img_byte, save_path)
#         FILE_QUEUE.put(save_path)

#     except Exception as e:
#         logger.error(f"Error in eval: {e}")


class Game_Agent(Agent):
    def __init__(
        self,
        model: str,
        script: str,
        tools: Union[ToolManager, List[Tool]],
        system_message: Optional[str] = None,
        access_token: str = "",
        max_round: int = 2,
    ) -> None:
        eb.api_type = "aistudio"
        eb.access_token = os.getenv("EB_ACCESS_TOKEN") if not access_token else access_token
        self.script = script
        memory = SlidingWindowMemory(max_round)
        super().__init__(
            llm=ERNIEBot(model), memory=memory, tools=tools, system_message=SystemMessage(system_message)
        )
        self.memory.msg_manager.messages = [
            HumanMessage(INSTRUCTION.format(SCRIPT=self.script)),
            AIMessage(content=f"好的，我将为你提供《{self.script}》沉浸式图文RPG场景体验。", function_call=None),
        ]

    def handle_tool(self, tool_name: str, tool_args: str) -> None:
        global FILE_QUEUE
        save_path = asyncio.run(
            self._async_run_tool(
                tool_name=tool_name,
                tool_args=tool_args,
            )
        )
        FILE_QUEUE.put(save_path)

    async def _async_run(self, prompt: str) -> AsyncGenerator[Any, Any]:
        """Defualt open stream for threading tool call

        Args:
        prompt: str, the prompt for the tool
        """

        actual_query = prompt + "根据我的选择继续生成一轮仅含包括<场景描述>、<场景图片>和<选择>的互动。"
        messages = self.memory.get_messages() + [HumanMessage(actual_query)]
        response = await self.llm.async_chat(messages, stream=True)

        apply = False
        res = ""
        function_part = None
        thread = None

        async for temp_res in response:
            for s in temp_res.content:
                # 用缓冲区来达成一个字一个字输出的流式
                res += s
                time.sleep(0.005)
                yield s, function_part, thread  # 将处理函数时需要用到的部分返回给外层函数

                if res.count("```") == 2 and not apply:  # TODO 判断逻辑待更改
                    function_part = res[res.find("```") : res.rfind("```") + 3]
                    tool = eval(function_part[function_part.find("{") : function_part.rfind("}") + 1])
                    # TODO：线程修改为异步函数
                    # loop = asyncio.get_running_loop()
                    # task = loop.create_task
                    thread = threading.Thread(
                        target=self.handle_tool, args=(tool["tool_name"], tool["tool_args"])
                    )
                    thread.start()
                    apply = True
        # await task

        self.memory.add_message(HumanMessage(prompt))
        self.memory.add_message(AIMessage(content=res, function_call=None))

    def reset_memory(self) -> None:
        self.memory.msg_manager.messages = [
            HumanMessage(INSTRUCTION.format(SCRIPT=self.script)),
            AIMessage(content=f"好的，我将为你提供《{self.script}》沉浸式图文RPG场景体验。", function_call=None),
        ]

    def launch_gradio_demo(self) -> Any:
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

    def _handle_gradio_chat(self, user_message, history) -> Tuple[str, List[tuple[str, str]]]:
        # 用于处理gradio的chatbot返回
        return "", history + [[user_message, None]]

    async def _handle_gradio_stream(self, history) -> AsyncGenerator:
        # 用于处理gradio的流式
        global FILE_QUEUE
        bot_response = self._async_run(history[-1][0])
        history[-1][1] = ""
        async for temp_response in bot_response:
            function_part = temp_response[1]
            thread = temp_response[2]
            history[-1][1] += temp_response[0]
            yield history
        else:
            if thread:
                thread.join()

            img_path = FILE_QUEUE.get().strip('"')  # 去除json.dump的引号
            logger.debug("end" + img_path)
            if function_part:
                history[-1][1] = history[-1][1].replace(
                    function_part,
                    f"<img src='file={img_path}' alt='Example Image' width='400' height='300'>",
                )
            yield history


if __name__ == "__main__":
    # from erniebot_agent.extensions.langchain.embeddings import ErnieEmbeddings
    # from langchain.embeddings import HuggingFaceEmbeddings
    # from langchain.vectorstores import FAISS
    # from erniebot_agent.tools.SearchTool import SearchTool

    # embeddings = ErnieEmbeddings(
    #     aistudio_access_token=os.environ.get('EB_ACCESS_TOKEN'),
    #     chunk_size=16,
    #     )

    # model_kwargs = {'device': 'mps'}
    # encode_kwargs = {'normalize_embeddings': True}
    # embeddings = HuggingFaceEmbeddings(
    #     model_name="shibing624/text2vec-base-chinese",
    #     model_kwargs=model_kwargs,
    #     # encode_kwargs=encode_kwargs,
    # )

    # db = FAISS.load_local(args.db_dir, embeddings)
    # searchtool = SearchTool(db)

    args = parse_args()
    game_system = Game_Agent(
        model=args.model,
        script=args.game,
        tools=[ImageGenerationTool()],
        system_message=SYSTEM_MESSAGE.format(SCRIPT=args.game),
    )
    game_system.launch_gradio_demo()
