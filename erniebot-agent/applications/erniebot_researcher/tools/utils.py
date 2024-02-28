import json
import logging
import os
import shutil
import urllib.parse
from typing import Any, Dict, List, Union

import jsonlines
import markdown  # type: ignore[import-untyped]
from langchain.docstore.document import Document
from langchain.output_parsers.json import parse_json_markdown
from weasyprint import CSS, HTML
from weasyprint.fonts import FontConfiguration

from erniebot_agent.agents.callback import LoggingHandler
from erniebot_agent.tools.base import BaseTool
from erniebot_agent.utils import config_from_environ as C
from erniebot_agent.utils.json import to_pretty_json
from erniebot_agent.utils.logging import ColorFormatter, set_role_color
from erniebot_agent.utils.output_style import ColoredContent

default_logger = logging.getLogger(__name__)


class ReportCallbackHandler(LoggingHandler):
    async def on_run_start(self, agent: Any, prompt, **kwargs):
        agent_name = kwargs.get("agent_name", None)
        if isinstance(prompt, (dict, list, tuple)):
            prompt = json.dumps(prompt, ensure_ascii=False)
        self._agent_info(
            "%s named %s is about to start running with input:\n%s",
            agent.__class__.__name__,
            agent_name,
            ColoredContent(prompt, role="user"),
            subject="Run",
            state="Start",
        )

    async def on_run_end(self, agent: Any, response, **kwargs):
        agent_name = kwargs.get("agent_name", None)
        self._agent_info(
            "%s %s finished running.", agent.__class__.__name__, agent_name, subject="Run", state="End"
        )

    async def on_tool_start(
        self, agent: Any, tool: Union[BaseTool, str], input_args: Union[str, Dict, List]
    ) -> None:
        if isinstance(input_args, (dict, list, tuple)):
            js_inputs = json.dumps(input_args, ensure_ascii=False)
        elif isinstance(input_args, str):
            js_inputs = input_args
        else:
            js_inputs = to_pretty_json(input_args, from_json=True)
        if isinstance(tool, BaseTool):
            tool_name = tool.__class__.__name__
        else:
            tool_name = tool
        self._agent_info(
            "%s is about to start running with input:\n%s",
            ColoredContent(tool_name, role="function"),
            ColoredContent(js_inputs, role="function"),
            subject="Tool",
            state="Start",
        )

    async def on_tool_end(self, agent: Any, tool: Union[BaseTool, str], response: Any) -> None:
        """Called to log when a tool ends running."""
        if isinstance(response, (dict, list, tuple)):
            js_inputs = json.dumps(response, ensure_ascii=False)
        else:
            js_inputs = to_pretty_json(response, from_json=True)
        if isinstance(tool, BaseTool):
            tool_name = tool.__class__.__name__
        else:
            tool_name = tool
        self._agent_info(
            "%s finished running with output:\n%s",
            ColoredContent(tool_name, role="function"),
            ColoredContent(js_inputs, role="function"),
            subject="Tool",
            state="End",
        )

    async def on_run_error(self, tool_name, error_information):
        self.logger.error(f"{tool_name}的调用失败，错误信息：{error_information}")

    def _agent_info(self, msg: str, *args, subject, state, **kwargs) -> None:
        msg = f"[{subject}][{state}] {msg}"
        self.logger.info(msg, *args, **kwargs)

    async def on_llm_error(self, agent: Any, llm, error):
        self.logger.error(f"LLM调用失败，错误信息:{error}")

    async def on_tool_error(self, agent: Any, tool, error):
        self.logger.error(f"Tool调用失败，错误信息:{error}")


def write_to_file(filename: str, text: str) -> None:
    """Write text to a file

    Args:
        text (str): The text to write
        filename (str): The filename to write to
    """
    with open(filename, "w") as file:
        file.write(text)


def convert_markdown_to_pdf(markdown_content: str, output_pdf_file: str):
    font_config = FontConfiguration()
    local_font_path = "SimSun.ttf"
    if not os.path.exists(local_font_path):
        raise RuntimeError("""SimSun.ttf not found, please download it""")
    local_font_path = os.path.abspath(local_font_path)
    css_str = f"""
        @font-face {{
            font-family: 'CustomFont';
            src: local('Custom Font'), url('file://{local_font_path}') format('truetype');
        }}
        body {{
            font-family: 'CustomFont';
        }}
    """
    css = CSS(string=css_str, font_config=font_config)
    html_content = markdown.markdown(markdown_content)
    HTML(string=html_content).write_pdf(output_pdf_file, stylesheets=[css], font_config=font_config)


def write_md_to_pdf(task: str, path: str, text: str) -> str:
    file_path = f"{path}/{task}"
    write_to_file(f"{file_path}.md", text)
    convert_markdown_to_pdf(text, f"{file_path}.pdf")
    encoded_file_path = urllib.parse.quote(f"{file_path}.pdf")
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


def add_citation(paragraphs, index_name, embeddings, build_index, SearchTool):
    if os.path.exists(index_name):
        shutil.rmtree(index_name)
    list_data = []
    for item in paragraphs:
        example = Document(page_content=item["summary"], metadata={"url": item["url"], "name": item["name"]})
        list_data.append(example)
    faiss_db = build_index(index_name=index_name, embeddings=embeddings, origin_data=list_data)
    faiss_search = SearchTool(db=faiss_db)
    return faiss_search


class JsonUtil:
    def parse_json(self, json_str, start_indicator: str = "{", end_indicator: str = "}"):
        if start_indicator == "{":
            response = parse_json_markdown(json_str)
        else:
            start_idx = json_str.index(start_indicator)
            end_idx = json_str.rindex(end_indicator)
            corrected_data = json_str[start_idx : end_idx + 1]
            response = json.loads(corrected_data)
        return response


def setup_logging(log_file_path: str):
    logger = logging.getLogger("generate_report")
    verbosity = C.get_logging_level()
    if verbosity:
        numeric_level = getattr(logging, verbosity.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid logging level: {verbosity}")
        logger.setLevel(numeric_level)
        logger.propagate = False
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColorFormatter("%(levelname)s - %(message)s"))
    logger.addHandler(console_handler)
    set_role_color()
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(ColorFormatter("%(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    return logger
