import os
import urllib.parse
from typing import Optional

import jsonlines
from erniebot_agent.retrieval.document import Document
from md2pdf.core import md2pdf

import erniebot

api_type = os.environ.get("api_type", None)
access_token = os.environ.get("access_token", None)


def erniebot_chat(messages: list, functions: Optional[str] = None, model: Optional[str] = None, **kwargs):
    if not model:
        model = "ernie-bot-4"
    _config = dict(
        api_type=api_type,
        access_token=access_token,
    )
    if functions is None:
        import time

        time.sleep(2)
        resp_stream = erniebot.ChatCompletion.create(
            _config_=_config, model=model, messages=messages, **kwargs, stream=False
        )
    else:
        resp_stream = erniebot.ChatCompletion.create(
            _config_=_config, model=model, messages=messages, **kwargs, functions=functions, stream=False
        )
    return resp_stream["result"]


def call_function(action: str, agent_role_prompt: str, model="ernie-bot-8k", **kwargs):
    messages = [
        {
            "role": "user",
            "content": action,
        }
    ]
    answer = erniebot_chat(messages, system=agent_role_prompt, model=model, **kwargs)
    return answer


def write_to_file(filename: str, text: str) -> None:
    """Write text to a file

    Args:
        text (str): The text to write
        filename (str): The filename to write to
    """
    with open(filename, "w") as file:
        file.write(text)


def md_to_pdf(input_file, output_file):
    md2pdf(output_file, md_content=None, md_file_path=input_file, css_file_path=None, base_url=None)


def write_md_to_pdf(task: str, path: str, text: str) -> str:
    file_path = f"{path}/{task}"
    write_to_file(f"{file_path}.md", text)

    # encoded_file_path = urllib.parse.quote(f"{file_path}.pdf")
    encoded_file_path = urllib.parse.quote(f"{file_path}.md")
    return encoded_file_path


def write_to_json(filename: str, list_data: list, mode="w") -> None:
    """Write text to a file

    Args:
        text (str): The text to write
        filename (str): The filename to write to
    """
    with jsonlines.open(filename, mode) as file:
        for item in list_data:
            file.write(item)


def json_correct(json_data):
    messages = [{"role": "user", "content": "请纠正以下数据的json格式：" + json_data}]
    res = erniebot_chat(messages)
    start_idx = res.index("{")
    end_idx = res.rindex("}")
    corrected_data = res[start_idx : end_idx + 1]
    return corrected_data


def add_citation(paragraphs, aurora_db):
    list_data = []
    for item in paragraphs:
        example = {"title": item["name"], "content_se": item["summary"]}
        example = Document.from_dict(example)
        example.meta["url"] = item["url"]
        list_data.append(example)
    res = aurora_db.add_documents(documents=list_data)
    return res
