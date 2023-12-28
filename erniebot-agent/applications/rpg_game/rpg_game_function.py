import argparse
import base64
import os
import tempfile
from typing import Any

from chat_story_tool import ChatStoryTool
from img_gen_tool import ImageGenerateTool

from erniebot_agent.agents import FunctionAgent
from erniebot_agent.agents.base import BaseAgent
from erniebot_agent.chat_models.erniebot import ERNIEBot
from erniebot_agent.memory import SlidingWindowMemory
from erniebot_agent.memory.messages import AIMessage, SystemMessage
from erniebot_agent.utils.common import get_file_type
from erniebot_agent.utils.html_format import IMAGE_HTML

os.environ["EB_AGENT_LOGGING_LEVEL"] = "info"

INSTRUCTION = """你的指令是为我提供一个基于《{SCRIPT}》剧情的在线RPG游戏体验。在这个游戏中，玩家将扮演《{SCRIPT}》剧情关键角色，游戏情景将基于《{SCRIPT}》剧情。\
这个游戏的玩法是互动式的，并遵循以下特定格式：

<场景描述>：根据玩家的选择，故事情节将按照《{SCRIPT}》剧情的线索发展。你将描述角色所处的环境和情况，不得少于三句话。

<场景图片>：对于每个场景，你将创造一个概括该场景情况的图像。

<选择>：在每次互动中，你将为玩家提供三个行动选项，分别标为1、2、3，以及第四个选项“输入玩家自定义的选择”。故事情节将根据玩家选择的行动进展。\
如果一个选择不是直接来自《{SCRIPT}》剧情，你将创造性地适应故事，最终引导它回归原始情节。

整个故事将围绕《{SCRIPT}》丰富而复杂的世界展开。每次互动必须包括<场景描述>、<场景图片>和<选择>。所有内容将以中文呈现。\
你的重点将仅仅放在提供场景描述，场景图片和选择上，不包含其他游戏指导。场景尽量不要重复，要丰富一些。

当我说游戏开始的时候，开始游戏。每次只要输出一组互动，不要自己生成互动。"""


def parse_args():
    parser = argparse.ArgumentParser(prog="erniebot-RPG")
    parser.add_argument("--access-token", type=str, default=None, help="Access token to use.")
    parser.add_argument("--game", type=str, default="射雕英雄传", help="Story name")
    parser.add_argument("--model", type=str, default="ernie-3.5", help="Model name")
    return parser.parse_args()


args = parse_args()


class GameAgent(FunctionAgent):
    def launch_gradio_demo(self: BaseAgent, **launch_kwargs: Any):
        try:
            import gradio as gr  # type: ignore
        except ImportError:
            raise ImportError(
                "Could not import gradio, which is required for `launch_gradio_demo()`."
                " Please run `pip install erniebot-agent[gradio]` to install the optional dependencies."
            ) from None

        raw_messages = []

        def _messages_to_dicts(messages):
            return [message.to_dict() for message in messages]

        def _pre_chat(text, history):
            history.append([text, None])
            return history, gr.update(value="", interactive=False), gr.update(interactive=False)

        async def _chat(history):
            prompt = history[-1][0]
            response = await self.run(prompt)
            self.memory.msg_manager.messages[-1] = AIMessage(
                eval(response.chat_history[2].content)["return_story"]
            )
            raw_messages.extend(response.chat_history)
            if len(response.chat_history) >= 3:
                output_result = eval(response.chat_history[2].content)["return_story"]
            else:
                output_result = response.text
            if response.steps and response.steps[-1].output_files:
                # If there is a file output in the last round, then we need to show it.
                output_file = response.steps[-1].output_files[-1]
                file_content = await output_file.read_contents()
                if get_file_type(output_file.filename) == "image":
                    # If it is a image, we can display it in the same chat page.
                    base64_encoded = base64.b64encode(file_content).decode("utf-8")
                    output_result = eval(response.chat_history[2].content)[
                        "return_story"
                    ] + IMAGE_HTML.format(BASE64_ENCODED=base64_encoded)
            history[-1][1] = output_result
            return (
                history,
                _messages_to_dicts(raw_messages),
                _messages_to_dicts(self.memory.get_messages()),
            )

        def _post_chat():
            return gr.update(interactive=True), gr.update(interactive=True)

        def _clear():
            raw_messages.clear()
            self.reset_memory()
            return None, None, None, None

        with gr.Blocks(
            title="ERNIE Bot Agent Demo", theme=gr.themes.Soft(spacing_size="sm", text_size="md")
        ) as demo:
            with gr.Column():
                with gr.Tab(label="Chat"):
                    chatbot = gr.Chatbot(
                        label="Chat history",
                        latex_delimiters=[
                            {"left": "$$", "right": "$$", "display": True},
                            {"left": "$", "right": "$", "display": False},
                        ],
                        bubble_full_width=False,
                        height=700,
                    )

                    with gr.Row():
                        prompt_textbox = gr.Textbox(
                            label="Prompt", placeholder="Write a prompt here...", scale=15
                        )
                        submit_button = gr.Button("Submit", min_width=150)
                        clear_button = gr.Button("Clear", min_width=100)

                    with gr.Accordion("Tools", open=False):
                        attached_tools = self.get_tools()
                        tool_descriptions = [tool.function_call_schema() for tool in attached_tools]
                        gr.JSON(value=tool_descriptions)
                    with gr.Accordion("Raw messages", open=False):
                        all_messages_json = gr.JSON(label="All messages")
                        agent_memory_json = gr.JSON(label="Messges in memory")

            prompt_textbox.submit(
                _pre_chat,
                inputs=[prompt_textbox, chatbot],
                outputs=[chatbot, prompt_textbox, submit_button],
            ).then(
                _chat,
                inputs=[chatbot],
                outputs=[
                    chatbot,
                    all_messages_json,
                    agent_memory_json,
                ],
            ).then(
                _post_chat, outputs=[prompt_textbox, submit_button]
            )
            submit_button.click(
                _pre_chat,
                inputs=[prompt_textbox, chatbot],
                outputs=[chatbot, prompt_textbox, submit_button],
            ).then(
                _chat,
                inputs=[chatbot],
                outputs=[
                    chatbot,
                    all_messages_json,
                    agent_memory_json,
                ],
            ).then(
                _post_chat, outputs=[prompt_textbox, submit_button]
            )
            clear_button.click(
                _clear,
                outputs=[
                    chatbot,
                    prompt_textbox,
                    all_messages_json,
                    agent_memory_json,
                ],
            )
        with tempfile.TemporaryDirectory() as td:
            if "allowed_paths" in launch_kwargs:
                if not isinstance(launch_kwargs["allowed_paths"], list):
                    raise TypeError("`allowed_paths` must be a list")
                allowed_paths = launch_kwargs["allowed_paths"] + [td]
                launch_kwargs.pop("allowed_paths")
            else:
                allowed_paths = [td]
            demo.launch(allowed_paths=allowed_paths, **launch_kwargs)


def creates_story_tool():
    memory = SlidingWindowMemory(max_round=2)
    llm = ERNIEBot(model=args.model, api_type="aistudio")
    agent = FunctionAgent(
        llm=llm, tools=[], system_message=SystemMessage(INSTRUCTION.format(SCRIPT=args.game)), memory=memory
    )
    tool = ChatStoryTool(agent, game=args.game)
    return tool


def main():
    img_tool = ImageGenerateTool()
    story_tool = creates_story_tool()
    SYSTEM_MESSAGE =  "你是《{SCRIPT}》沉浸式图文RPG场景助手，能够生成图文剧情。\
                    每次用户发送query或者输入数字开始互动时，\
                    请你先调用ChatStoryTool生成互动，然后调用ImageGenerateTool生成图片，\
                    最后输出的时候回答'已完成'即可。"

    llm = ERNIEBot(model=args.model, api_type="aistudio", enable_multi_step_tool_call=True)
    memory = SlidingWindowMemory(max_round=2)
    agent = GameAgent(
        llm=llm,
        tools=[story_tool, img_tool],
        memory=memory,
        system_message=SystemMessage(SYSTEM_MESSAGE.format(SCRIPT=args.game)),
    )
    agent.launch_gradio_demo()


if __name__ == "__main__":
    main()
