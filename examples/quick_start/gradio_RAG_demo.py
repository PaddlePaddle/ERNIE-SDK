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

import time
import math
import erniebot
import faiss
import os
import numpy as np
import gradio as gr

from tqdm import tqdm
from typing import List, Union

CSS = """
    #col {
        width: min(100%, 800px);
        top: 0;
        right: 0;
        bottom: 0;
        left: 0;
        margin: auto;
    }
    
    footer{display:none !important}
"""

REF_HTML = """

<details style="border: 1px solid #ccc; padding: 10px; border-radius: 4px; margin-bottom: 4px">
    <summary style="display: flex; align-items: center; font-weight: bold;">
        <span style="margin-right: 10px;">[{index}] {title}</span>
        <a style="text-decoration: none; background: none !important;" target="_blank">
            <!--[Here should be a link icon]-->
            <i style="border: solid #000; border-width: 0 2px 2px 0; display: inline-block; padding: 3px; transform:rotate(-45deg); -webkit-transform(-45deg)"></i>   
        </a>
    </summary>
    <p style="margin-top: 10px;">{text}</p>
</details>

"""

PROMPT_TEMPLATE = "基于以下已知信息，请简洁并专业地回答用户的问题。如果无法从中得到答案，请说 '根据已知信息无法回答该问题' 或 '没有提供足够的相关信息'。不允许在答案中添加编造成分。你可以参考以下文章:\n{DOCS}\n问题：{QUERY}\n回答："

def split_by_len(texts:List[str], split_token:int =384) -> List[str]:
    split_token = split_token # 文心支持最大长度
    idx = 0
    chunk = []

    for text in texts:
        while idx + split_token < len(text):
            temp_text = text[idx:idx + split_token]
            next_idx = temp_text.rfind('。') + 1
            if next_idx != 0: # 如果该切片中没有句号，则直接添加整个句子
                chunk.append(temp_text[:next_idx])
                idx = idx + next_idx
            else:
                chunk.append(temp_text)
                idx = idx + split_token

        chunk.append(text[idx:])
    
    return chunk

def get_embedding(word: List[str]) -> List[float]:
    """
    获取单词的embedding向量

    Args:
        word (List[str]): 需要获取embedding向量的单词列表

    Returns:
        List[float]: 单词的embedding向量列表
    """
    if len(word) <= 16:
        embedding = erniebot.Embedding.create(
                                            model = 'ernie-text-embedding',
                                            input = word
                                            ).get_result()
    else:
        size = len(word)
        embedding = []
        for i in tqdm(range(math.ceil(size / 16))):
            embedding.extend(erniebot.Embedding.create(model = 'ernie-text-embedding', input = word[i*16:(i+1)*16]).get_result())
            time.sleep(1)
    return embedding

def l2_normalization(embedding:np.ndarray) -> np.ndarray:
    if embedding.ndim == 1:
        return embedding / np.linalg.norm(embedding).reshape(-1,1)
    else:
        return embedding/np.linalg.norm(embedding,axis=1).reshape(-1,1)


def find_related_doc(
        query:str, 
        origin_chunk:List[str], 
        index_ip:faiss.swigfaiss.IndexFlatIP, 
        top_k:int=5
    ) -> Union[str, List[int]]:
    
    D, I = index_ip.search(np.array(get_embedding([query])),top_k)
    top_k_similar = I.tolist()[0]

    res = ''
    ref_lis = []
    for i in range(top_k):
        res += f"[参考文章{i+1}]:{origin_chunk[top_k_similar[i]]}" + '\n\n'
        ref_lis.append(origin_chunk[top_k_similar[i]])
    return res, ref_lis

def process_uploaded_file(files):
    content = []
    
    for file in files:
        with open(file, 'r') as f:
            content.append(f.read())

    doc_chunk = split_by_len(content)

    doc_embedding = get_embedding(doc_chunk)
    assert len(doc_embedding) == len(doc_chunk),"shape mismatch"
    doc_embedding = l2_normalization(np.array(doc_embedding))

    index_ip = faiss.IndexFlatIP(doc_embedding.shape[1])
    index_ip.add(doc_embedding)

    temp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)

    faiss.write_index(index_ip, os.path.join(temp_path, 'knowledge_embedding.index'))
    with open(os.path.join(temp_path, 'knowledge.txt'),'w') as f:
        for chunk in doc_chunk:
            f.write(repr(chunk) + '\n')

    return '已完成向量知识库搭建'


def get_ans(query: str):

    temp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    doc_chunk = []
    with open(os.path.join(temp_path, 'knowledge.txt'),'r') as f:
        for line in f:
            doc_chunk.append(eval(line))
    index_ip = faiss.read_index(os.path.join(temp_path, 'knowledge_embedding.index')) 
    related_doc, references = find_related_doc(query,doc_chunk,index_ip)
    
    refs = []
    for i in range(len(references)):
        temp_dict = {
            'title': f'Reference{i+1}',
            'text' : references[i],
        }
        refs.append(temp_dict)

    answer = erniebot.ChatCompletion.create(
        model='ernie-bot-4', 
        messages=[{'role': 'user', 'content': PROMPT_TEMPLATE.format(DOCS=related_doc, QUERY=query)}]).get_result()
        
    return answer, "<h3>References (Click to Expand)</h3>" + "\n".join([REF_HTML.format(**item, index = idx + 1) for idx, item in enumerate(refs)])
   

with gr.Blocks(theme=gr.themes.Base(), css=CSS) as demo:
    gr.Markdown("# 文心大模型RAG问答DEMO")
    with gr.Tabs():
        with gr.TabItem("对话栏"):
            with gr.Row():
                query_box = gr.Textbox(show_label=False, placeholder="Enter question and press ENTER")
            
            answer_box = gr.Textbox(show_label=False, value='', lines=5)
            ref_boxes = gr.HTML(label="References")

        with gr.TabItem("设置栏"):
            with gr.Row():
                with gr.Column():  # 左边一列是输入
                    file_upload = gr.Files(file_types=['txt'], label='目前仅支持txt格式文件')
                    chat_box = gr.Textbox(show_label=False)
                with gr.Column():  # 右边一列是输出
                    ernie_model = gr.Dropdown(
                        label="Model", info="模型类型", value="ernie-bot", choices=["ernie-bot", "ernie-bot-turbo", "ernie-bot-4"], interactive=True
                    )
                    api_type = gr.Dropdown(
                        label="API Type", info="提供对话能力的后端平台", value="qianfan", choices=["qianfan", "aistudio"], interactive=True
                    )
                    access_token = gr.Textbox(
                        label="Access Token", info="用于访问后端平台的access token，如果选择aistudio，则需设置此参数", type="password"
                    )
                    access_key = gr.Textbox(
                        label="AK", info="用于访问千帆平台的AK，如果选择qianfan，则需设置此参数", type="password"
                    )
                    secret_key = gr.Textbox(
                        label="SK", info="用于访问千帆平台的SK，如果选择qianfan，则需设置此参数", type="password"
                    )
                    top_p = gr.Slider(
                        label="Top-p", info="控制采样范围，该参数越小生成结果越稳定", value=0.7, minimum=0, maximum=1, step=0.05, interactive=True
                    )
                    temperature = gr.Slider(
                        label="Temperature",
                        info="控制采样随机性，该参数越小生成结果越稳定",
                        value=0.95,
                        minimum=0.05,
                        maximum=1,
                        step=0.05,
                        interactive=True,
                    )
                    _config = {
                        'ernie_model':ernie_model,
                        'api_type':api_type,
                        'AK':access_key,
                        'SK':secret_key,
                        'access_token':access_token,
                        'top_p':top_p,
                        'temperature':temperature
                    }
                    
    query_box.submit(get_ans, query_box, [answer_box, ref_boxes])
    file_upload.upload(process_uploaded_file, file_upload, chat_box)
    erniebot.api_type = 'aistudio'
    erniebot.access_token = 'd86186382de8cceb4512efbd774b74ea72f3a9f5'
if __name__ == "__main__":
    demo.launch()