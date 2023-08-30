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
import copy
import gradio as gr
import requests
import json
import os
import time
import numpy as np
import erniebot as eb


def setup_args():
    """Setup arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8073)
    args = parser.parse_args()
    return args


def launch(args):
    """Launch characters dialogue demo."""

    def rollback(state):
        """Rollback context."""
        context = state.setdefault("context", [])
        content = context[-2]["content"]
        context = context[:-2]
        state["context"] = context
        shown_context = get_shown_context(context)
        return content, shown_context, context, state

    def cosine_similarity(vec_0, vec_1):
        dot_result = float(np.dot(vec_0, vec_1))
        denom = np.linalg.norm(vec_0) * np.linalg.norm(vec_1)
        return 0.5 + 0.5 * (dot_result / denom) if denom != 0 else 0

    def cal_embedding(text1, text2, api_type, access_key, secret_key):
        eb.api_type = api_type
        eb.ak = access_key
        eb.sk = secret_key
        if access_key.strip() == "" or secret_key.strip() == "":
            raise gr.exceptions.Error("需要填写正确的AK/SK，不能为空")

        if text1.strip() == "" or text2.strip() == "":
            raise gr.exceptions.Error("两个输入均不能为空")
        embeddings = eb.Embedding.create(
            model="ernie-text-embedding", input=[text1.strip(), text2.strip()])
        emb_0 = embeddings.body["data"][0]["embedding"]
        emb_1 = embeddings.body["data"][1]["embedding"]
        cos_sim = cosine_similarity(emb_0, emb_1)
        cos_sim_text = "## 两段文本余弦相似度: {}".format(cos_sim)
        return "{}".format(emb_0), "{}".format(emb_1), cos_sim_text

    def gen_image(prompt, w_and_h, api_type, access_key, secret_key):
        timestamp = int(time.time())
        eb.api_type = api_type
        eb.ak = access_key
        eb.sk = secret_key
        if access_key.strip() == "" or secret_key.strip() == "":
            raise gr.exceptions.Error("需要填写正确的AK/SK，不能为空")
        if prompt.strip() == "":
            raise gr.exceptions.Error("输入不能为空")
        w, h = [int(x) for x in w_and_h.strip().split("x")]
        response = eb.Image.create(
            model='ernie-vilg-v2',
            prompt=prompt,
            width=w,
            height=h,
            version='v2',
            image_num=1)
        img_url = response.data['sub_task_result_list'][0]['final_image_list'][
            0]['img_url']
        res = requests.get(img_url)
        with open("{}.jpg".format(timestamp), "wb") as f:
            f.write(res.content)
        return "{}.jpg".format(timestamp)

    def regen(ernie_model, state, top_p, temperature, max_length, api_type,
              access_key, secret_key):
        """Regenerate response."""
        context = state.setdefault("context", [])
        context.pop()
        user_turn = context.pop()
        return infer(ernie_model, user_turn["content"], state, top_p,
                     temperature, max_length, api_type, access_key, secret_key)

    def infer(ernie_model, content, state, top_p, temperature, max_length,
              api_type, access_key, secret_key):
        """Model inference."""
        eb.api_type = api_type
        eb.ak = access_key
        eb.sk = secret_key
        if access_key.strip() == "" or secret_key.strip() == "":
            raise gr.exceptions.Error("需要填写正确的AK/SK，不能为空, 清空后重试")
        if content.strip() == "":
            raise gr.exceptions.Error("输入不能为空，清空后重试")
        content = content.strip().replace("<br>", "\n")
        context = state.setdefault("context", [])
        context.append({"role": "user", "content": content})
        data = {
            "context": content,
            "top_p": top_p,
            "temperature": temperature,
            "max_length": max_length,
            "min_length": 1,
        }
        model = "ernie-bot-3.5" if ernie_model is None or ernie_model.strip(
        ) == "" else ernie_model
        if ernie_model == "chat_file":
            response = eb.ChatFile.create(messages=context, stream=False)
        else:
            response = eb.ChatCompletion.create(
                model=model, messages=context, stream=False)

        bot_response = response.result
        context.append({"role": "assistant", "content": bot_response})
        shown_context = get_shown_context(context)
        return None, shown_context, context, state

    def clean_context(context):
        """Clean context for EB input."""
        cleaned_context = copy.deepcopy(context)
        for turn in cleaned_context:
            if turn["role"] == "bot":
                bot_resp = turn["content"]
                if bot_resp.startswith("<img src") or bot_resp.startswith(
                        "<audio controls>"):
                    bot_resp = "\n".join(bot_resp.split("\n")[1:])
                turn["content"] = bot_resp
        return cleaned_context

    def get_shown_context(context):
        """Get gradio chatbot."""
        shown_context = []
        for turn_idx in range(0, len(context), 2):
            shown_context.append([
                context[turn_idx]["content"], context[turn_idx + 1]["content"]
            ])

        return shown_context

    with gr.Blocks(title="ERNIE Bot SDK", theme=gr.themes.Soft()) as block:
        with gr.Row(scale=2):
            gr.Markdown("# ERNIE Bot SDK")
        with gr.Tab("ChatCompletion") as chat_completion_tab:
            with gr.Row():
                with gr.Column(scale=1):
                    api_type = gr.Dropdown(
                        choices=["qianfan"],
                        label="Api Type",
                        value="qianfan",
                        info="选择对话能力的提供平台")
                    access_key = gr.Textbox(
                        placeholder="Access Key ID",
                        label="Access Key ID",
                        type="password",
                        info="用于访问对话能力平台的AK")
                    secret_key = gr.Textbox(
                        placeholder="Secret Access Key",
                        label="Secret Access Key",
                        type="password",
                        info="用于访问对话能力平台的SK")
                    ernie_model = gr.Dropdown(
                        choices=["ernie-bot-3.5", "ernie-bot-turbo"],
                        label="Model",
                        value="ernie-bot-3.5",
                        info="模型类型，默认ernie-bot-3.5")
                    top_p = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=0.7,
                        step=0.05,
                        label="Top-p",
                        info="控制生成环节采样范围，参数越小，生成结果越稳定")
                    temperature = gr.Slider(
                        minimum=0.05,
                        maximum=1.5,
                        value=0.95,
                        step=0.05,
                        label="Temperature",
                        info="控制生成环境采样随机性，参数越小，生成结果越稳定")
                    max_length = gr.Slider(
                        minimum=1,
                        maximum=1024,
                        value=10,
                        step=1,
                        label="Max Length",
                        info="生成Token的最大数量设置")
                with gr.Column(scale=4):
                    state = gr.State({})
                    context_chatbot = gr.Chatbot(
                        label="Context", scroll_to_output=True)
                    utt_text = gr.Textbox(placeholder="请输入...", label="Content")
                    with gr.Row():
                        clear_btn = gr.Button("清空")
                        rollback_btn = gr.Button("撤回")
                        regen_btn = gr.Button("重新生成")
                        send_btn = gr.Button("发送")
                    with gr.Row():
                        raw_context_json = gr.JSON(label="Raw context")

                chat_completion_tab.select(
                    lambda _: (None, None, None, {}),
                    outputs=[
                        utt_text, context_chatbot, raw_context_json, state
                    ])
                utt_text.submit(
                    infer,
                    inputs=[
                        ernie_model, utt_text, state, top_p, temperature,
                        max_length, api_type, access_key, secret_key
                    ],
                    outputs=[
                        utt_text, context_chatbot, raw_context_json, state
                    ],
                    api_name="chat",
                )
                clear_btn.click(
                    lambda _: (None, None, None, {}),
                    inputs=clear_btn,
                    outputs=[
                        utt_text, context_chatbot, raw_context_json, state
                    ],
                    api_name="clear",
                    show_progress=False,
                )
                rollback_btn.click(
                    rollback,
                    inputs=[state],
                    outputs=[
                        utt_text, context_chatbot, raw_context_json, state
                    ],
                    show_progress=False,
                )
                regen_btn.click(
                    regen,
                    inputs=[
                        ernie_model, state, top_p, temperature, max_length,
                        api_type, access_key, secret_key
                    ],
                    outputs=[
                        utt_text, context_chatbot, raw_context_json, state
                    ],
                )
                send_btn.click(
                    infer,
                    inputs=[
                        ernie_model, utt_text, state, top_p, temperature,
                        max_length, api_type, access_key, secret_key
                    ],
                    outputs=[
                        utt_text, context_chatbot, raw_context_json, state
                    ],
                    api_name="chat")

        with gr.Tab("Embedding"):
            gr.Markdown("输入两段文本，将会分别计算得到两段文本的向量表示，同时计算两段文本的余弦相似度")
            with gr.Row(scale=1):
                with gr.Column(scale=1):
                    api_type = gr.Dropdown(
                        choices=["qianfan"],
                        label="Api Type",
                        value="qianfan",
                        info="选择Embedding能力的提供平台")
                    access_key = gr.Textbox(
                        placeholder="Access Key ID",
                        label="Access Key ID",
                        type="password",
                        info="用于访问云服务平台的AK")
                    secret_key = gr.Textbox(
                        placeholder="Secret Access Key",
                        label="Secret Access Key",
                        type="password",
                        info="用于访问云服务平台的SK")
                with gr.Column(scale=4):
                    with gr.Row(scale=4):
                        text1 = gr.Textbox(placeholder="输入第一段文本", label="第一段文本")
                        text2 = gr.Textbox(placeholder="输入第二段文本", label="第二段文本")
                    cal_emb = gr.Button("计算向量")
                    cos_sim = gr.Markdown("## 两段文本余弦相似度: -")
                    with gr.Row():
                        embedding1 = gr.Textbox(label="文本1向量结果")
                        embedding2 = gr.Textbox(label="文本2向量结果")
                    cal_emb.click(
                        cal_embedding,
                        inputs=[
                            text1, text2, api_type, access_key, secret_key
                        ],
                        outputs=[embedding1, embedding2, cos_sim])

        with gr.Tab("Image"):
            with gr.Row():
                with gr.Column(scale=1):
                    api_type = gr.Dropdown(
                        choices=["yinian"],
                        label="Api Type",
                        value="yinian",
                        info="选择文生图能力的提供平台")
                    access_key = gr.Textbox(
                        placeholder="Access Key ID",
                        label="Access Key ID",
                        type="password",
                        info="用于访问云服务平台的AK")
                    secret_key = gr.Textbox(
                        placeholder="Secret Access Key",
                        label="Secret Access Key",
                        type="password",
                        info="用于访问云服务平台的SK")
                with gr.Column(scale=4):
                    with gr.Row():
                        prompt = gr.Textbox(
                            placeholder="输入你的Prompt用于生成图片，例如： 生成一朵玫瑰花",
                            label="Prompt")
                        w_and_h = gr.Dropdown(
                            choices=[
                                "512x512", "640x360", "360x640", "1024x1024",
                                "1280x720", "720x1280", "2048x2048",
                                "2560x1440", "1440x2560"
                            ],
                            value="512x512",
                            label="分辨率")
                    submit_button = gr.Button("点击生成图片")
                    image_show_zone = gr.Image(
                        label="图片生成结果",
                        type="filepath",
                        show_download_button=True)
                    submit_button.click(
                        gen_image,
                        inputs=[
                            prompt, w_and_h, api_type, access_key, secret_key
                        ],
                        outputs=image_show_zone)
    block.launch(server_name="0.0.0.0", server_port=args.port, debug=True)


def main(args):
    launch(args)


if __name__ == "__main__":
    args = setup_args()
    main(args)
