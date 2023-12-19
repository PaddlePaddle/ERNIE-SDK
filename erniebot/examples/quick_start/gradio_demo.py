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
import math
import os
import time
from collections.abc import Iterator
from typing import List

import faiss
import gradio as gr
import numpy as np
import requests
from tqdm import tqdm

import erniebot as eb


def parse_setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8073)
    args = parser.parse_args()
    return args


def create_ui_and_launch(args):
    with gr.Blocks(title="ERNIE Bot SDK Demos", theme=gr.themes.Soft()) as blocks:
        gr.Markdown("# ERNIE Bot SDK基础功能演示")
        create_chat_completion_tab()
        create_embedding_tab()
        create_image_tab()
        create_rag_tab()

    blocks.launch(server_name="0.0.0.0", server_port=args.port)


def create_chat_completion_tab():
    def _infer(
        ernie_model, content, state, top_p, temperature, api_type, access_key, secret_key, access_token
    ):
        access_key = access_key.strip()
        secret_key = secret_key.strip()
        access_token = access_token.strip()

        if (access_key == "" or secret_key == "") and access_token == "":
            raise gr.Error("需要填写正确的AK/SK或access token，不能为空")
        if content.strip() == "":
            raise gr.Error("输入不能为空，请在清空后重试")

        auth_config = {
            "api_type": api_type,
        }
        if access_key:
            auth_config["ak"] = access_key
        if secret_key:
            auth_config["sk"] = secret_key
        if access_token:
            auth_config["access_token"] = access_token

        content = content.strip().replace("<br>", "")
        context = state.setdefault("context", [])
        context.append({"role": "user", "content": content})
        data = {
            "messages": context,
            "top_p": top_p,
            "temperature": temperature,
        }

        if ernie_model == "chat_file":
            response = eb.ChatFile.create(_config_=auth_config, **data, stream=False)
        else:
            response = eb.ChatCompletion.create(
                _config_=auth_config, model=ernie_model, **data, stream=False
            )

        bot_response = response.result
        context.append({"role": "assistant", "content": bot_response})
        history = _get_history(context)
        return None, history, context, state

    def _regen_response(
        ernie_model, state, top_p, temperature, api_type, access_key, secret_key, access_token
    ):
        """Regenerate response."""
        context = state.setdefault("context", [])
        if len(context) < 2:
            raise gr.Error("请至少进行一轮对话")
        context.pop()
        user_message = context.pop()
        return _infer(
            ernie_model,
            user_message["content"],
            state,
            top_p,
            temperature,
            api_type,
            access_key,
            secret_key,
            access_token,
        )

    def _rollback(state):
        """Roll back context."""
        context = state.setdefault("context", [])
        content = context[-2]["content"]
        context = context[:-2]
        state["context"] = context
        history = _get_history(context)
        return content, history, context, state

    def _get_history(context):
        history = []
        for turn_idx in range(0, len(context), 2):
            history.append([context[turn_idx]["content"], context[turn_idx + 1]["content"]])

        return history

    with gr.Tab("对话补全（Chat Completion）") as chat_completion_tab:
        with gr.Row():
            with gr.Column(scale=1):
                api_type = gr.Dropdown(
                    label="API Type", info="提供对话能力的后端平台", value="qianfan", choices=["qianfan", "aistudio"]
                )
                access_key = gr.Textbox(
                    label="AK", info="用于访问后端平台的AK，如果设置了access token则无需设置此参数", type="password"
                )
                secret_key = gr.Textbox(
                    label="SK", info="用于访问后端平台的SK，如果设置了access token则无需设置此参数", type="password"
                )
                access_token = gr.Textbox(
                    label="Access Token", info="用于访问后端平台的access token，如果设置了AK、SK则无需设置此参数", type="password"
                )
                ernie_model = gr.Dropdown(
                    label="Model", info="模型类型", value="ernie-bot", choices=["ernie-bot", "ernie-bot-turbo"]
                )
                top_p = gr.Slider(
                    label="Top-p", info="控制采样范围，该参数越小生成结果越稳定", value=0.7, minimum=0, maximum=1, step=0.05
                )
                temperature = gr.Slider(
                    label="Temperature",
                    info="控制采样随机性，该参数越小生成结果越稳定",
                    value=0.95,
                    minimum=0.05,
                    maximum=1,
                    step=0.05,
                )
            with gr.Column(scale=4):
                state = gr.State({})
                context_chatbot = gr.Chatbot(label="对话历史")
                input_text = gr.Textbox(label="消息内容", placeholder="请输入...")
                with gr.Row():
                    clear_btn = gr.Button("清空")
                    rollback_btn = gr.Button("撤回")
                    regen_btn = gr.Button("重新生成")
                    send_btn = gr.Button("发送")
                raw_context_json = gr.JSON(label="原始对话上下文信息")

        api_type.change(
            lambda api_type: {
                "qianfan": (gr.update(visible=True), gr.update(visible=True)),
                "aistudio": (gr.update(visible=False), gr.update(visible=False)),
            }[api_type],
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
    def _get_embeddings(text1, text2, api_type, access_key, secret_key, access_token):
        access_key = access_key.strip()
        secret_key = secret_key.strip()
        access_token = access_token.strip()

        if (access_key == "" or secret_key == "") and access_token == "":
            raise gr.Error("需要填写正确的AK/SK或access token，不能为空")

        auth_config = {
            "api_type": api_type,
        }
        if access_key:
            auth_config["ak"] = access_key
        if secret_key:
            auth_config["sk"] = secret_key
        if access_token:
            auth_config["access_token"] = access_token

        if text1.strip() == "" or text2.strip() == "":
            raise gr.Error("两个输入均不能为空")
        embeddings = eb.Embedding.create(
            _config_=auth_config,
            model="ernie-text-embedding",
            input=[text1.strip(), text2.strip()],
        )
        emb_0 = embeddings.rbody["data"][0]["embedding"]
        emb_1 = embeddings.rbody["data"][1]["embedding"]
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
                    label="API Type", info="提供语义向量能力的后端平台", value="qianfan", choices=["qianfan", "aistudio"]
                )
                access_key = gr.Textbox(
                    label="AK", info="用于访问后端平台的AK，如果设置了access token则无需设置此参数", type="password"
                )
                secret_key = gr.Textbox(
                    label="SK", info="用于访问后端平台的SK，如果设置了access token则无需设置此参数", type="password"
                )
                access_token = gr.Textbox(
                    label="Access Token", info="用于访问后端平台的access token，如果设置了AK、SK则无需设置此参数", type="password"
                )
            with gr.Column(scale=4):
                with gr.Row():
                    text1 = gr.Textbox(label="第一段文本", placeholder="输入第一段文本")
                    text2 = gr.Textbox(label="第二段文本", placeholder="输入第二段文本")
                cal_emb = gr.Button("生成向量")
                cos_sim = gr.Markdown("## 余弦相似度: -")
                with gr.Row():
                    embedding1 = gr.Textbox(label="文本1向量结果")
                    embedding2 = gr.Textbox(label="文本2向量结果")

        api_type.change(
            lambda api_type: {
                "qianfan": (gr.update(visible=True), gr.update(visible=True)),
                "aistudio": (gr.update(visible=False), gr.update(visible=False)),
            }[api_type],
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
    def _gen_image(prompt, w_and_h, api_type, access_key, secret_key, access_token):
        access_key = access_key.strip()
        secret_key = secret_key.strip()
        access_token = access_token.strip()

        if (access_key == "" or secret_key == "") and access_token == "":
            raise gr.Error("需要填写正确的AK/SK或access token，不能为空")
        if prompt.strip() == "":
            raise gr.Error("输入不能为空")

        auth_config = {
            "api_type": api_type,
        }
        if access_key:
            auth_config["ak"] = access_key
        if secret_key:
            auth_config["sk"] = secret_key
        if access_token:
            auth_config["access_token"] = access_token

        timestamp = int(time.time())
        w, h = [int(x) for x in w_and_h.strip().split("x")]

        response = eb.Image.create(
            _config_=auth_config,
            model="ernie-vilg-v2",
            prompt=prompt,
            width=w,
            height=h,
            version="v2",
            image_num=1,
        )
        img_url = response.data["sub_task_result_list"][0]["final_image_list"][0]["img_url"]
        res = requests.get(img_url)
        with open(f"{timestamp}.jpg", "wb") as f:
            f.write(res.content)
        return f"{timestamp}.jpg"

    with gr.Tab("文生图（Image Generation）"):
        with gr.Row():
            with gr.Column(scale=1):
                api_type = gr.Dropdown(
                    label="API Type", info="提供文生图能力的后端平台", value="yinian", choices=["yinian"]
                )
                access_key = gr.Textbox(
                    label="AK", info="用于访问后端平台的AK，如果设置了access token则无需设置此参数", type="password"
                )
                secret_key = gr.Textbox(
                    label="SK", info="用于访问后端平台的SK，如果设置了access token则无需设置此参数", type="password"
                )
                access_token = gr.Textbox(
                    label="Access Token", info="用于访问后端平台的access token，如果设置了AK、SK则无需设置此参数", type="password"
                )
            with gr.Column(scale=4):
                with gr.Row():
                    prompt = gr.Textbox(label="Prompt", placeholder="输入用于生成图片的prompt，例如： 生成一朵玫瑰花")
                    w_and_h = gr.Dropdown(
                        label="分辨率",
                        value="512x512",
                        choices=[
                            "512x512",
                            "640x360",
                            "360x640",
                            "1024x1024",
                            "1280x720",
                            "720x1280",
                            "2048x2048",
                            "2560x1440",
                            "1440x2560",
                        ],
                    )
                submit_btn = gr.Button("生成图片")
                image_show_zone = gr.Image(label="图片生成结果", type="filepath", show_download_button=True)

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


def create_rag_tab():
    REF_HTML = """

    <details style="border: 1px solid #ccc; padding: 10px; border-radius: 4px; margin-bottom: 4px">
        <summary style="display: flex; align-items: center; font-weight: bold;">
            <span style="margin-right: 10px;">[{index}] {title}</span>
            <a style="text-decoration: none; background: none !important;" target="_blank">
                <!--[Here should be a link icon]-->
                <i style="border: solid #000; border-width: 0 2px 2px 0; display: inline-block; padding: 3px;
                transform:rotate(-45deg);-webkit-transform(-45deg)">
                </i>
            </a>
        </summary>
        <p style="margin-top: 10px;">{text}</p>
    </details>

    """

    PROMPT_TEMPLATE = """基于以下已知信息，请简洁并专业地回答用户的问题。
如果无法从中得到答案，请说 '根据已知信息无法回答该问题' 或 '没有提供足够的相关信息'。不允许在答案中添加编造成分。
你可以参考以下文章:
{DOCS}
问题：{QUERY}
回答："""

    _CONFIG = {
        "ernie_model": "",
        "api_type": "",
        "AK": "",
        "SK": "",
        "access_token": "",
        "top_p": 0.7,
        "temperature": 0.95,
    }

    def split_by_len(texts: List[str], split_token: int = 384) -> List[str]:
        """
        Split the knowledge base docs into chunks by length.

        Args:
            texts (List[str]): Knowledge Base Texts.
            split_token (int, optional): The max length supported by ernie-text-embedding. Default to 384.

        Returns:
            List[str]: Doc Chunks.
        """
        chunk = []
        for text in texts:
            idx = 0
            while idx + split_token < len(text):
                temp_text = text[idx : idx + split_token]
                next_idx = temp_text.rfind("。") + 1
                if next_idx != 0:  # If this slice doesn't have a period, add the whole sentence.
                    chunk.append(temp_text[:next_idx])
                    idx = idx + next_idx
                else:
                    chunk.append(temp_text)
                    idx = idx + split_token

            chunk.append(text[idx:])
        return chunk

    def _get_embedding_doc(word: List[str]) -> List[float]:
        """
        Get the embedding of a list of words.

        Args:
            word (List[str]): Words to get embedding.

        Returns:
            List[float]: Embedding List of the words.
        """
        if (_CONFIG["AK"] == "" or _CONFIG["SK"] == "") and _CONFIG["access_token"] == "":
            raise gr.Error("需要填写正确的AK/SK或access token，不能为空")

        embedding: List[float]
        if len(word) <= 16:
            resp = eb.Embedding.create(model="ernie-text-embedding", input=word)
            assert not isinstance(resp, Iterator)
            embedding = resp.get_result()
        else:
            size = len(word)
            embedding = []
            for i in tqdm(range(math.ceil(size / 16))):
                temp_result = eb.Embedding.create(
                    model="ernie-text-embedding", input=word[i * 16 : (i + 1) * 16]
                )
                assert not isinstance(temp_result, Iterator)
                embedding.extend(temp_result.get_result())
                time.sleep(1)
        return embedding

    def l2_normalization(embedding: np.ndarray) -> np.ndarray:
        "Vector Normalization by l2 norm"
        if embedding.ndim == 1:
            return embedding / np.linalg.norm(embedding).reshape(-1, 1)
        else:
            return embedding / np.linalg.norm(embedding, axis=1).reshape(-1, 1)

    def find_related_doc(
        query: str, origin_chunk: List[str], index_ip: faiss.swigfaiss.IndexFlatIP, top_k: int = 5
    ) -> tuple[str, List[int]]:
        """
        Fin top_k similar documents.

        Args:
            query (str): user query.
            origin_chunk (List[str]): Knowledge Base Doc.
            index_ip (faiss.swigfaiss.IndexFlatIP): Vector DB index。
            top_k (int, optional): Return top_k most similar documents. Default to 5.

        Returns:
            str, List[int]: The most similar documents and their index.
        """

        D, Idx = index_ip.search(np.array(_get_embedding_doc([query])), top_k)
        top_k_similar = Idx.tolist()[0]

        res = ""
        ref_lis = []
        for i in range(top_k):
            res += f"[参考文章{i+1}]:{origin_chunk[top_k_similar[i]]}" + "\n\n"
            ref_lis.append(origin_chunk[top_k_similar[i]])
        return res, ref_lis

    def process_uploaded_file(files: List[str], *args: object) -> str:
        """
        Args:
            files: Files path
            _CONFIG: Config
        """
        _update_config(*args)

        content = []
        for file in files:
            with open(file, "r") as f:
                content.append(f.read())

        doc_chunk = split_by_len(content)

        doc_embedding = _get_embedding_doc(doc_chunk)
        assert len(doc_embedding) == len(doc_chunk), "shape mismatch"
        doc_embedding_arr = l2_normalization(np.array(doc_embedding))

        index_ip = faiss.IndexFlatIP(doc_embedding_arr.shape[1])
        index_ip.add(doc_embedding_arr)

        temp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)

        faiss.write_index(index_ip, os.path.join(temp_path, "knowledge_embedding.index"))
        with open(os.path.join(temp_path, "knowledge.txt"), "w") as f:
            for chunk in doc_chunk:
                f.write(repr(chunk) + "\n")

        return "已完成向量知识库搭建"

    def get_ans(query: str, *args: object) -> tuple[str, str]:
        _update_config(*args)

        if (_CONFIG["AK"] == "" or _CONFIG["SK"] == "") and _CONFIG["access_token"] == "":
            raise gr.Error("需要填写正确的AK/SK或access token，不能为空")
        temp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        doc_chunk = []
        with open(os.path.join(temp_path, "knowledge.txt"), "r") as f:
            for line in f:
                doc_chunk.append(eval(line))
        index_ip = faiss.read_index(os.path.join(temp_path, "knowledge_embedding.index"))
        related_doc, references = find_related_doc(query, doc_chunk, index_ip)

        refs = []
        for i in range(len(references)):
            temp_dict = {
                "title": f"Reference{i+1}",
                "text": references[i],
            }
            refs.append(temp_dict)

        resp = eb.ChatCompletion.create(
            model=_CONFIG["ernie_model"],
            messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(DOCS=related_doc, QUERY=query)}],
            top_p=_CONFIG["top_p"],
            temperature=_CONFIG["temperature"],
        )
        assert not isinstance(resp, Iterator)
        answer = resp.get_result()

        return answer, "<h3>References (Click to Expand)</h3>" + "\n".join(
            [REF_HTML.format(**item, index=idx + 1) for idx, item in enumerate(refs)]
        )

    def _update_config(*args: object):
        eb.api_type = args[1]
        eb.access_token = args[2]
        eb.AK = args[3]
        eb.SK = args[4]

        _CONFIG.update(
            {
                "ernie_model": args[0],
                "api_type": args[1],
                "access_token": args[2],
                "AK": args[3],
                "SK": args[4],
                "top_p": args[5],
                "temperature": args[6],
            }
        )
        # print(_CONFIG)

    with gr.Tab("知识库问答(Retrieval Augmented QA)"):
        # gr.Markdown("# 文心大模型RAG问答DEMO")
        with gr.Tabs():
            with gr.TabItem("设置栏"):
                with gr.Row():
                    with gr.Column():
                        file_upload = gr.Files(file_types=["txt"], label="目前仅支持txt格式文件")
                        chat_box = gr.Textbox(show_label=False)
                    with gr.Column():
                        ernie_model = gr.Dropdown(
                            label="Model",
                            info="模型类型",
                            value="ernie-bot-4",
                            choices=["ernie-bot-4", "ernie-bot-turbo", "ernie-bot"],
                        )
                        api_type = gr.Dropdown(
                            label="API Type",
                            info="提供对话能力的后端平台",
                            value="aistudio",
                            choices=["aistudio", "qianfan"],
                        )
                        access_token = gr.Textbox(
                            label="Access Token",
                            info="用于访问后端平台的access token，如果选择aistudio，则需设置此参数",
                            type="password",
                        )
                        access_key = gr.Textbox(
                            label="AK", info="用于访问千帆平台的AK，如果选择qianfan，则需设置此参数", type="password"
                        )
                        secret_key = gr.Textbox(
                            label="SK", info="用于访问千帆平台的SK，如果选择qianfan，则需设置此参数", type="password"
                        )
                        top_p = gr.Slider(
                            label="Top-p",
                            info="控制采样范围，该参数越小生成结果越稳定",
                            value=0.7,
                            step=0.05,
                            minimum=0,
                            maximum=1,
                        )
                        temperature = gr.Slider(
                            label="temperature",
                            info="控制采样随机性，该参数越小生成结果越稳定",
                            value=0.95,
                            step=0.05,
                            maximum=1,
                            minimum=0,
                        )

            with gr.TabItem("问答栏"):
                with gr.Row():
                    query_box = gr.Textbox(show_label=False, placeholder="Enter question and press ENTER")

                answer_box = gr.Textbox(show_label=False, value="", lines=5)
                ref_boxes = gr.HTML(label="References")

        query_box.submit(
            get_ans,
            [query_box, ernie_model, api_type, access_token, access_key, secret_key, top_p, temperature],
            [answer_box, ref_boxes],
        )
        file_upload.upload(
            process_uploaded_file,
            [file_upload, ernie_model, api_type, access_token, access_key, secret_key, top_p, temperature],
            chat_box,
        )


if __name__ == "__main__":
    args = parse_setup_args()
    create_ui_and_launch(args)
