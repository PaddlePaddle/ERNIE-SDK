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
import sys
major = sys.version_info.major
minor = sys.version_info.minor
if int(major) != 3 or int(minor) < 8:
    raise RuntimeError(
        f"The Gradio demo requires Python >= 3.8, but your Python version is {major}.{minor}."
    )
import time

import erniebot as eb
import gradio as gr
import numpy as np
import requests


def parse_setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8073)
    args = parser.parse_args()
    return args


def create_ui_and_launch(args):
    with gr.Blocks(title="ERNIE Bot SDK", theme=gr.themes.Soft()) as block:
        with gr.Row():
            gr.Markdown("# ERNIE Bot SDK")
        create_chat_completion_tab()
        create_embedding_tab()
        create_image_tab()

    block.launch(server_name="0.0.0.0", server_port=args.port)


def create_chat_completion_tab():
    def _infer(ernie_model, content, state, top_p, temperature, api_type,
               access_key, secret_key, access_token):
        access_key = access_key.strip()
        secret_key = secret_key.strip()
        access_token = access_token.strip()

        if (access_key == '' or secret_key == '') and access_token == '':
            raise gr.Error("需要填写正确的AK/SK或access token，不能为空")
        if content.strip() == '':
            raise gr.Error("输入不能为空，请在清空后重试")

        auth_config = {'api_type': api_type, }
        if access_key:
            auth_config['ak'] = access_key
        if secret_key:
            auth_config['sk'] = secret_key
        if access_token:
            auth_config['access_token'] = access_token

        content = content.strip().replace('<br>', '\n')
        context = state.setdefault('context', [])
        context.append({'role': 'user', 'content': content})
        data = {
            'messages': context,
            'top_p': top_p,
            'temperature': temperature,
        }

        if ernie_model == 'chat_file':
            response = eb.ChatFile.create(
                _config_=auth_config, **data, stream=False)
        else:
            response = eb.ChatCompletion.create(
                _config_=auth_config, model=ernie_model, **data, stream=False)

        bot_response = response.result
        context.append({'role': 'assistant', 'content': bot_response})
        history = _get_history(context)
        return None, history, context, state

    def _regen_response(ernie_model, state, top_p, temperature, api_type,
                        access_key, secret_key, access_token):
        """Regenerate response."""
        context = state.setdefault('context', [])
        if len(context) < 2:
            raise gr.Error("请至少进行一轮对话")
        context.pop()
        user_message = context.pop()
        return _infer(ernie_model, user_message['content'], state, top_p,
                      temperature, api_type, access_key, secret_key,
                      access_token)

    def _rollback(state):
        """Roll back context."""
        context = state.setdefault('context', [])
        content = context[-2]['content']
        context = context[:-2]
        state['context'] = context
        history = _get_history(context)
        return content, history, context, state

    def _get_history(context):
        history = []
        for turn_idx in range(0, len(context), 2):
            history.append([
                context[turn_idx]['content'], context[turn_idx + 1]['content']
            ])

        return history

    with gr.Tab("对话补全（Chat Completion）") as chat_completion_tab:
        with gr.Row():
            with gr.Column(scale=1):
                api_type = gr.Dropdown(
                    label="API Type",
                    info="提供对话能力的后端平台",
                    value='qianfan',
                    choices=['qianfan', 'aistudio'])
                access_key = gr.Textbox(
                    label="AK",
                    info="用于访问后端平台的AK，如果设置了access token则无需设置此参数",
                    type='password')
                secret_key = gr.Textbox(
                    label="SK",
                    info="用于访问后端平台的SK，如果设置了access token则无需设置此参数",
                    type='password')
                access_token = gr.Textbox(
                    label="Access Token",
                    info="用于访问后端平台的access token，如果设置了AK、SK则无需设置此参数",
                    type='password')
                ernie_model = gr.Dropdown(
                    label="Model",
                    info="模型类型",
                    value='ernie-bot',
                    choices=['ernie-bot', 'ernie-bot-turbo'])
                top_p = gr.Slider(
                    label="Top-p",
                    info="控制采样范围，该参数越小生成结果越稳定",
                    value=0.7,
                    minimum=0,
                    maximum=1,
                    step=0.05)
                temperature = gr.Slider(
                    label="Temperature",
                    info="控制采样随机性，该参数越小生成结果越稳定",
                    value=0.95,
                    minimum=0.05,
                    maximum=1,
                    step=0.05)
            with gr.Column(scale=4):
                state = gr.State({})
                context_chatbot = gr.Chatbot(label="对话历史")
                input_text = gr.Textbox(label="消息内容", placeholder="请输入...")
                with gr.Row():
                    clear_btn = gr.Button("清空")
                    rollback_btn = gr.Button("撤回")
                    regen_btn = gr.Button("重新生成")
                    send_btn = gr.Button("发送")
                with gr.Row():
                    raw_context_json = gr.JSON(label="原始对话上下文信息")

        api_type.change(
            lambda api_type: {
                'qianfan': (gr.update(visible=True), gr.update(visible=True)),
                'aistudio': (gr.update(visible=False), gr.update(visible=False)), }[api_type],
            inputs=api_type,
            outputs=[
                access_key,
                secret_key,
            ],
        )
        chat_completion_tab.select(
            lambda: (None, None, None, {}),
            outputs=[
                input_text,
                context_chatbot,
                raw_context_json,
                state,
            ],
        )
        input_text.submit(
            _infer,
            inputs=[
                ernie_model,
                input_text,
                state,
                top_p,
                temperature,
                api_type,
                access_key,
                secret_key,
                access_token,
            ],
            outputs=[
                input_text,
                context_chatbot,
                raw_context_json,
                state,
            ],
        )
        clear_btn.click(
            lambda _: (None, None, None, {}),
            inputs=clear_btn,
            outputs=[
                input_text,
                context_chatbot,
                raw_context_json,
                state,
            ],
            show_progress=False,
        )
        rollback_btn.click(
            _rollback,
            inputs=[state],
            outputs=[
                input_text,
                context_chatbot,
                raw_context_json,
                state,
            ],
            show_progress=False,
        )
        regen_btn.click(
            _regen_response,
            inputs=[
                ernie_model,
                state,
                top_p,
                temperature,
                api_type,
                access_key,
                secret_key,
                access_token,
            ],
            outputs=[
                input_text,
                context_chatbot,
                raw_context_json,
                state,
            ],
        )
        send_btn.click(
            _infer,
            inputs=[
                ernie_model,
                input_text,
                state,
                top_p,
                temperature,
                api_type,
                access_key,
                secret_key,
                access_token,
            ],
            outputs=[
                input_text,
                context_chatbot,
                raw_context_json,
                state,
            ],
        )


def create_embedding_tab():
    def _get_embeddings(text1, text2, api_type, access_key, secret_key,
                        access_token):
        access_key = access_key.strip()
        secret_key = secret_key.strip()
        access_token = access_token.strip()

        if (access_key == '' or secret_key == '') and access_token == '':
            raise gr.Error("需要填写正确的AK/SK或access token，不能为空")

        auth_config = {'api_type': api_type, }
        if access_key:
            auth_config['ak'] = access_key
        if secret_key:
            auth_config['sk'] = secret_key
        if access_token:
            auth_config['access_token'] = access_token

        if text1.strip() == '' or text2.strip() == '':
            raise gr.Error("两个输入均不能为空")
        embeddings = eb.Embedding.create(
            _config_=auth_config,
            model='ernie-text-embedding',
            input=[text1.strip(), text2.strip()],
        )
        emb_0 = embeddings.rbody['data'][0]['embedding']
        emb_1 = embeddings.rbody['data'][1]['embedding']
        cos_sim = _calc_cosine_similarity(emb_0, emb_1)
        cos_sim_text = f"## 两段文本余弦相似度: {cos_sim}"
        return str(emb_0), str(emb_1), cos_sim_text

    def _calc_cosine_similarity(vec_0, vec_1):
        dot_result = float(np.dot(vec_0, vec_1))
        denom = np.linalg.norm(vec_0) * np.linalg.norm(vec_1)
        return 0.5 + 0.5 * (dot_result / denom) if denom != 0 else 0

    with gr.Tab("语义向量（Embedding）"):
        gr.Markdown("输入两段文本，分别获取两段文本的向量表示，并计算向量间的余弦相似度")
        with gr.Row():
            with gr.Column(scale=1):
                api_type = gr.Dropdown(
                    label="API Type",
                    info="提供语义向量能力的后端平台",
                    value='qianfan',
                    choices=['qianfan', 'aistudio'])
                access_key = gr.Textbox(
                    label="AK",
                    info="用于访问后端平台的AK，如果设置了access token则无需设置此参数",
                    type='password')
                secret_key = gr.Textbox(
                    label="SK",
                    info="用于访问后端平台的SK，如果设置了access token则无需设置此参数",
                    type='password')
                access_token = gr.Textbox(
                    label="Access Token",
                    info="用于访问后端平台的access token，如果设置了AK、SK则无需设置此参数",
                    type='password')
            with gr.Column(scale=4):
                with gr.Row():
                    text1 = gr.Textbox(label="第一段文本", placeholder="输入第一段文本")
                    text2 = gr.Textbox(label="第二段文本", placeholder="输入第二段文本")
                cal_emb = gr.Button("提取向量")
                cos_sim = gr.Markdown("## 余弦相似度: -")
                with gr.Row():
                    embedding1 = gr.Textbox(label="文本1向量结果")
                    embedding2 = gr.Textbox(label="文本2向量结果")

        api_type.change(
            lambda api_type: {
                'qianfan': (gr.update(visible=True), gr.update(visible=True)),
                'aistudio': (gr.update(visible=False), gr.update(visible=False)), }[api_type],
            inputs=api_type,
            outputs=[
                access_key,
                secret_key,
            ],
        )
        cal_emb.click(
            _get_embeddings,
            inputs=[
                text1,
                text2,
                api_type,
                access_key,
                secret_key,
                access_token,
            ],
            outputs=[
                embedding1,
                embedding2,
                cos_sim,
            ],
        )


def create_image_tab():
    def _gen_image(prompt, w_and_h, api_type, access_key, secret_key,
                   access_token):
        access_key = access_key.strip()
        secret_key = secret_key.strip()
        access_token = access_token.strip()

        if (access_key == '' or secret_key == '') and access_token == '':
            raise gr.Error("需要填写正确的AK/SK或access token，不能为空")
        if prompt.strip() == '':
            raise gr.Error("输入不能为空")

        auth_config = {'api_type': api_type, }
        if access_key:
            auth_config['ak'] = access_key
        if secret_key:
            auth_config['sk'] = secret_key
        if access_token:
            auth_config['access_token'] = access_token

        timestamp = int(time.time())
        w, h = [int(x) for x in w_and_h.strip().split('x')]

        response = eb.Image.create(
            _config_=auth_config,
            model='ernie-vilg-v2',
            prompt=prompt,
            width=w,
            height=h,
            version='v2',
            image_num=1,
        )
        img_url = response.data['sub_task_result_list'][0]['final_image_list'][
            0]['img_url']
        res = requests.get(img_url)
        with open(f"{timestamp}.jpg", 'wb') as f:
            f.write(res.content)
        return f"{timestamp}.jpg"

    with gr.Tab("文生图（Image Generation）"):
        with gr.Row():
            with gr.Column(scale=1):
                api_type = gr.Dropdown(
                    label="API Type",
                    info="提供文生图能力的后端平台",
                    value='yinian',
                    choices=['yinian'])
                access_key = gr.Textbox(
                    label="AK",
                    info="用于访问后端平台的AK，如果设置了access token则无需设置此参数",
                    type='password')
                secret_key = gr.Textbox(
                    label="SK",
                    info="用于访问后端平台的SK，如果设置了access token则无需设置此参数",
                    type='password')
                access_token = gr.Textbox(
                    label="Access Token",
                    info="用于访问后端平台的access token，如果设置了AK、SK则无需设置此参数",
                    type='password')
            with gr.Column(scale=4):
                with gr.Row():
                    prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="输入用于生成图片的prompt，例如： 生成一朵玫瑰花")
                    w_and_h = gr.Dropdown(
                        label="分辨率",
                        value='512x512',
                        choices=[
                            '512x512', '640x360', '360x640', '1024x1024',
                            '1280x720', '720x1280', '2048x2048', '2560x1440',
                            '1440x2560'
                        ])
                submit_btn = gr.Button("生成图片")
                image_show_zone = gr.Image(
                    label="图片生成结果", type='filepath', show_download_button=True)

        submit_btn.click(
            _gen_image,
            inputs=[
                prompt,
                w_and_h,
                api_type,
                access_key,
                secret_key,
                access_token,
            ],
            outputs=image_show_zone,
        )


if __name__ == '__main__':
    args = parse_setup_args()
    create_ui_and_launch(args)
