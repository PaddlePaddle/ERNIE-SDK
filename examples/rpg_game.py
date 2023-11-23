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

import os
import sys
import argparse
import erniebot
import platform

from IPython.display import clear_output as clear

INSTRUCTION = """你的指令是为我提供一个基于《{SCRIPT}》剧情的在线RPG游戏体验。在这个游戏中，玩家将扮演《{SCRIPT}》剧情关键角色，游戏情景将基于《{SCRIPT}》剧情。这个游戏的玩法是互动式的，并遵循以下特定格式：

<场景描述>：根据玩家的选择，故事情节将按照《{SCRIPT}》剧情的线索发展。你将描述角色所处的环境和情况。

<场景图片>：对于每个场景，你将创造一个概括该情况的图像。这些图像的风格将类似于1980年代RPG游戏对话，16:9宽屏比例。在这个步骤你需要调用画图工具。

<选择>：在每次互动中，你将为玩家提供三个行动选项，分别标为1、2、3，以及第四个选项“输入玩家自定义的选择”。故事情节将根据玩家选择的行动进展。如果一个选择不是直接来自《{SCRIPT}》剧情，你将创造性地适应故事，最终引导它回归原始情节。

整个故事将围绕金《{SCRIPT}》丰富而复杂的世界展开。每次互动必须包括<场景描述>、<场景图片>和<选择>。所有内容将以中文呈现。你的重点将仅仅放在提供场景描述，场景图片和选择上，不包含其他游戏指导。场景尽量不要重复，要丰富一些。

当我说游戏开始的时候，开始游戏。当我说重新开始的时候，则重新开始整个游戏。"""

def parse_args():
    parser = argparse.ArgumentParser(prog="erniebot-RPG")
    parser.add_argument("--access-token", type=str, help="Access token to use.")
    parser.add_argument("--game", type=str, default='仙剑奇侠传',help="story name")
    parser.add_argument("--model", type=str, default='ernie-bot-4',help="Model name")
    return parser.parse_args()


def _clear_screen():
    os.system("cls" if platform.system() == "Windows" else "clear")
    if 'ipykernel' in sys.modules:
        clear()

class RPGGame:
    def __init__(
            self,
            model: str,
            script: str,
            access_token: str = None
    ) -> None:
        
        self.model = model
        self.script = script        
        self.chat_history = [
            {'role': 'user', 'content': INSTRUCTION.format(SCRIPT = self.script)},
            {'role': 'assistant', 'content': f"好的，我将为你提供《{self.script}》沉浸式图文RPG场景体验。"},
            ]

        erniebot.api_type = 'aistudio'
        erniebot.access_token = os.getenv("EB_ACCESS_TOKEN") if not access_token else access_token 

    def chat(
        self,query: str
    ) -> str:
        "Use this function to chat with ERNIE BOT"
        self.chat_history.append({'role': 'user', 'content': query})
        response = erniebot.ChatCompletion.create(
                model=self.model, 
                messages=self.chat_history,
                system=f"你是《{self.script}》沉浸式图文RPG场景助手，能够生成图文剧情，并给出玩家选项，整个故事将围绕《{self.script}》丰富而复杂的世界展开。每次互动必须包括<场景描述>、<场景图片>和<选择>。",
                )
        self.chat_history.append({'role': 'assistant', 'content': response.get_result()})
        return response.get_result()

    def chat_stream(
        self,query:str
    ) -> None:
        "Use this function to chat with ERNIE BOT"
        self.chat_history.append({'role': 'user', 'content': query})
        response = erniebot.ChatCompletion.create(
                model=self.model, 
                messages=self.chat_history,
                stream=True
                )
        result = ""

        for resp in response:
            result += resp.get_result()
            _clear_screen()
            print(result,flush=True)
        self.chat_history.append({'role': 'assistant', 'content': result})
        
       
    def clear(self) -> None:
        self.chat_history = [
            {'role': 'user', 'content': INSTRUCTION.format(SCRIPT = self.script)},
            {'role': 'assistant', 'content': '好的，我将为你提供《仙剑奇侠传》沉浸式图文RPG场景体验。'},
            ]
    
    def lauch_gradio(self) -> None:
        import gradio as gr
        with gr.Blocks() as demo:
            context_chatbot = gr.Chatbot(label="对话历史", height=600)
            input_text = gr.Textbox(label="消息内容", placeholder="请输入...")

            with gr.Row():
                start_buttton = gr.Button("开始游戏")
                remake_buttton = gr.Button("重新开始")
            
            remake_buttton.click(self.clear)
            start_buttton.click(self._gradio_chat, start_buttton, [input_text, context_chatbot])
            input_text.submit(self._gradio_chat, input_text, [input_text, context_chatbot])

        demo.launch()
    
    def _gradio_chat(self, query: str) -> tuple[str, str]:
        self.chat(query)
        history = []
        
        for i in range(2, len(self.chat_history), 2):
            history.append(
                (self.chat_history[i]['content'], self.chat_history[i+1]['content'])
            )

        return '', history
                
        
if __name__ == '__main__':
    args = parse_args()
    game_system = RPGGame(model=args.model, script=args.game)
    game_system.lauch_gradio()


                   
        