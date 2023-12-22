import base64
import os
import tempfile
from typing import Any, List

from erniebot_agent.file.base import File
from erniebot_agent.file.file_manager import FileManager
from erniebot_agent.tools.tool_manager import ToolManager
from erniebot_agent.utils.common import get_file_type
from erniebot_agent.utils.html_format import IMAGE_HTML, ITEM_LIST_HTML


class GradioMixin:
    _file_manager: FileManager  # make mypy happy
    _tool_manager: ToolManager  # make mypy happy

    def launch_gradio_demo(self, **launch_kwargs: Any):
        # TODO: Unified optional dependencies management
        try:
            import gradio as gr  # type: ignore
        except ImportError:
            raise ImportError(
                "Could not import gradio, which is required for `launch_gradio_demo()`."
                " Please run `pip install erniebot-agent[gradio]` to install the optional dependencies."
            ) from None

        raw_messages = []
        _uploaded_file_cache: List[File] = []

        def _pre_chat(text, history):
            history.append([text, None])
            return history, gr.update(value="", interactive=False), gr.update(interactive=False)

        async def _chat(history):
            nonlocal _uploaded_file_cache, td
            temp_save_file_name = None

            prompt = history[-1][0]
            if len(prompt) == 0:
                raise gr.Error("Prompt should not be empty.")

            if _uploaded_file_cache:
                response = await self.async_run(prompt, files=_uploaded_file_cache)
                _uploaded_file_cache = []
            else:
                response = await self.async_run(prompt)

            if (
                response.files
                and response.files[-1].type == "output"
                and response.files[-1].used_by == response.actions[-1].tool_name
            ):
                # If there is a file output in the last round, then we need to show it
                output_file_id = response.files[-1].file.id
                output_file = self._file_manager.look_up_file_by_id(output_file_id)
                file_content = await output_file.read_contents()
                if get_file_type(response.files[-1].file.filename) == "image":
                    # If it is a image, we can display it in the same chat page
                    base64_encoded = base64.b64encode(file_content).decode("utf-8")
                    if output_file_id in response.text:
                        output_result = response.text
                        output_result = output_result.replace(
                            output_file_id, IMAGE_HTML.format(BASE64_ENCODED=base64_encoded)
                        )
                    else:
                        output_result = response.text + IMAGE_HTML.format(BASE64_ENCODED=base64_encoded)
                else:
                    # TODO: Support multiple files, support audio now
                    temp_save_file_name = os.path.join(td, response.files[-1].file.filename)
                    with open(temp_save_file_name, "wb") as f:
                        f.write(file_content)
                    output_result = response.text

            else:
                output_result = response.text

            history[-1][1] = output_result
            if temp_save_file_name:
                # If it is not image, we should have another chat page
                history = history + [(None, (temp_save_file_name,))]
            raw_messages.extend(response.chat_history)
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

        async def _upload(file: List[gr.utils.NamedString], history: list):
            nonlocal _uploaded_file_cache
            for single_file in file:
                upload_file = await self._file_manager.create_file_from_path(single_file.name)
                _uploaded_file_cache.append(upload_file)
                history = history + [((single_file.name,), None)]
            size = len(file)

            output_lis = self._file_manager.registry.list_files()
            item = ""
            for i in range(len(output_lis) - size):
                item += f'<li>{str(output_lis[i]).strip("<>")}</li>'

            # The file uploaded this time will be gathered and colored
            item += "<li>"
            for i in range(size, 0, -1):
                item += f'{str(output_lis[len(output_lis)-i]).strip("<>")}<br>'
            item += "</li>"

            return ITEM_LIST_HTML.format(ITEM=item), history

        def _messages_to_dicts(messages):
            return [message.to_dict() for message in messages]

        with gr.Blocks(
            title="ERNIE Bot Agent Demo", theme=gr.themes.Soft(spacing_size="sm", text_size="md")
        ) as demo:
            with gr.Column():
                chatbot = gr.Chatbot(
                    label="Chat history",
                    latex_delimiters=[
                        {"left": "$$", "right": "$$", "display": True},
                        {"left": "$", "right": "$", "display": False},
                    ],
                    bubble_full_width=False,
                    height=750,
                )

                with gr.Row():
                    prompt_textbox = gr.Textbox(
                        label="Prompt", placeholder="Write a prompt here...", scale=15
                    )
                    submit_button = gr.Button("Submit", min_width=150)
                    with gr.Column(min_width=100):
                        clear_button = gr.Button("Clear", min_width=100)
                        file_button = gr.UploadButton(
                            "Upload",
                            min_width=100,
                            file_count="multiple",
                            file_types=["image", "video", "audio"],
                        )

                with gr.Accordion("Files", open=False):
                    file_lis = self._file_manager.registry.list_files()
                    all_files = gr.HTML(value=file_lis, label="All input files")
                with gr.Accordion("Tools", open=False):
                    attached_tools = self._tool_manager.get_tools()
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
            file_button.upload(
                _upload,
                inputs=[file_button, chatbot],
                outputs=[all_files, chatbot],
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
