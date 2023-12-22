from __future__ import annotations

import base64
import json
from copy import deepcopy
from typing import Any, Dict, List, Optional, Type

import requests

from erniebot_agent.file import GlobalFileManagerHandler
from erniebot_agent.file.file_manager import FileManager
from erniebot_agent.memory.messages import Message
from erniebot_agent.tools.base import BaseTool
from erniebot_agent.tools.schema import RemoteToolView
from erniebot_agent.tools.utils import (
    get_file_info_from_param_view,
    parse_file_from_json_response,
    parse_file_from_response,
    tool_response_contains_file,
)
from erniebot_agent.utils.common import is_json_response
from erniebot_agent.utils.exceptions import RemoteToolError
from erniebot_agent.utils.logging import logger


def check_json_length(value: Dict[str, Any]):
    """check the dict contains base64 string

    Args:
        value (Dict[str, Any]): the source of json data
    """
    json_string = json.dumps(value)
    if len(json_string) > 4096:
        raise RemoteToolError(
            "The length of json response is greater than 4096, please "
            "check whether `format:byte` is missing in openapi.yaml or "
            "the tool returned too much information.",
            stage="Output parsing",
        )


class RemoteTool(BaseTool):
    def __init__(
        self,
        tool_view: RemoteToolView,
        server_url: str,
        headers: dict,
        version: str,
        file_manager: Optional[FileManager],
        examples: Optional[List[Message]] = None,
        tool_name_prefix: Optional[str] = None,
    ) -> None:
        self.tool_view = tool_view
        self.server_url = server_url
        self.headers = headers
        self.version = version
        self.file_manager = file_manager
        self._examples = examples
        self.tool_name_prefix = tool_name_prefix
        # If `tool_name_prefix`` is provided, we prepend `tool_name_prefix`` to the `name` field of all tools
        if tool_name_prefix is not None and not self.tool_view.name.startswith(f"{self.tool_name_prefix}/"):
            self.tool_view.name = f"{self.tool_name_prefix}/{self.tool_view.name}"

        self.response_prompt: Optional[str] = None

    @property
    def examples(self) -> List[Message]:
        return self._examples or []

    def __str__(self) -> str:
        return "<name: {0}, server_url: {1}, description: {2}>".format(
            self.tool_name, self.server_url, self.tool_view.description
        )

    def __repr__(self):
        return self.__str__()

    @property
    def tool_name(self):
        return self.tool_view.name

    async def __pre_process__(self, tool_arguments: Dict[str, Any]) -> dict:
        async def fileid_to_byte(file_id, file_manager):
            file = file_manager.look_up_file_by_id(file_id)
            byte_str = await file.read_contents()
            return byte_str

        async def convert_to_file_data(file_data: str, format: str):
            value = file_data.replace("<file>", "").replace("</file>", "")
            byte_value = await fileid_to_byte(value, file_manager)
            if format == "byte":
                byte_value = base64.b64encode(byte_value).decode()
            return byte_value

        file_manager = await self._get_file_manager()

        # 1. replace fileid with byte string
        parameter_file_info = get_file_info_from_param_view(self.tool_view.parameters)
        for key in tool_arguments.keys():
            if self.tool_view.parameters:
                if key not in self.tool_view.parameters.model_fields:
                    keys = list(self.tool_view.parameters.model_fields.keys())
                    raise RemoteToolError(
                        f"`{self.tool_name}` received unexpected arguments `{key}`. "
                        f"The avaiable arguments are {keys}",
                        stage="Input parsing",
                    )
            if key not in parameter_file_info:
                continue
            if self.tool_view.parameters is None:
                break

            argument_value = tool_arguments[key]
            if isinstance(argument_value, list):
                for index in range(len(argument_value)):
                    argument_value[index] = await convert_to_file_data(
                        argument_value[index], parameter_file_info[key]["format"]
                    )
            else:
                argument_value = await convert_to_file_data(
                    argument_value, parameter_file_info[key]["format"]
                )

            tool_arguments[key] = argument_value

        # 2. call tool get response
        if self.tool_view.parameters is not None:
            tool_arguments = dict(self.tool_view.parameters(**tool_arguments))

        return tool_arguments

    async def __post_process__(self, tool_response: dict) -> dict:
        tool_response = self.__adhoc_post_process__(tool_response)
        check_json_length(tool_response)
        if self.response_prompt is not None:
            tool_response["prompt"] = self.response_prompt
        elif self.tool_view.returns is not None and self.tool_view.returns.__prompt__ is not None:
            tool_response["prompt"] = self.tool_view.returns.__prompt__
        elif tool_response_contains_file(tool_response):
            tool_response["prompt"] = (
                "参考工具说明中对各个结果字段的描述，提取工具调用结果中的信息，生成一段通顺的文本满足用户的需求。"
                "请务必确保每个符合'file-'格式的字段只出现一次，无需将其转换为链接，也无需添加任何HTML、Markdown或其他格式化元素。"
            )

        # TODO(wj-Mcat): open the tool-response valdiation with pydantic model
        # if self.tool_view.returns is not None:
        #     tool_response = dict(self.tool_view.returns(**tool_response))
        return tool_response

    async def __call__(self, **tool_arguments: Dict[str, Any]) -> Any:
        tool_arguments = await self.__pre_process__(tool_arguments)
        tool_response = await self.send_request(tool_arguments)
        return await self.__post_process__(tool_response)

    async def send_request(self, tool_arguments: Dict[str, Any]) -> dict:
        url = self.server_url + self.tool_view.uri + "?version=" + self.version

        headers = deepcopy(self.headers)
        headers["Content-Type"] = self.tool_view.parameters_content_type

        requests_inputs = {
            "headers": headers,
        }
        if self.tool_view.method == "get":
            requests_inputs["params"] = tool_arguments
        elif self.tool_view.parameters_content_type == "application/json":
            requests_inputs["json"] = tool_arguments
        elif self.tool_view.parameters_content_type in [
            "application/x-www-form-urlencoded",
        ]:
            requests_inputs["data"] = tool_arguments
        elif self.tool_view.parameters_content_type == "multipart/form-data":
            parameter_file_infos = get_file_info_from_param_view(self.tool_view.parameters)
            requests_inputs["files"] = {}
            for file_key in parameter_file_infos.keys():
                if file_key in tool_arguments:
                    requests_inputs["files"][file_key] = tool_arguments.pop(file_key)
                    headers.pop("Content-Type", None)
            requests_inputs["data"] = tool_arguments
        else:
            raise RemoteToolError(
                f"Unsupported content type: {self.tool_view.parameters_content_type}", stage="Executing"
            )
        if self.tool_view.method == "get":
            response = requests.get(url, **requests_inputs)  # type: ignore
        elif self.tool_view.method == "post":
            response = requests.post(url, **requests_inputs)  # type: ignore
        elif self.tool_view.method == "put":
            response = requests.put(url, **requests_inputs)  # type: ignore
        elif self.tool_view.method == "delete":
            response = requests.delete(url, **requests_inputs)  # type: ignore
        else:
            raise RemoteToolError(f"method<{self.tool_view.method}> is invalid", stage="Executing")

        if response.status_code != 200:
            logger.debug(f"The resource requested returned the following headers: {response.headers}")
            raise RemoteToolError(
                f"The resource requested by `{self.tool_name}` "
                f"returned {response.status_code}: {response.text}",
                stage="Executing",
            )

        # parse the file from response
        returns_file_infos = get_file_info_from_param_view(self.tool_view.returns)

        if len(returns_file_infos) == 0 and is_json_response(response):
            return response.json()

        file_manager = await self._get_file_manager()

        file_metadata = {"tool_name": self.tool_name}
        if is_json_response(response) and len(returns_file_infos) > 0:
            response_json = response.json()
            file_info = await parse_file_from_json_response(
                response_json,
                file_manager=file_manager,
                param_view=self.tool_view.returns,  # type: ignore
                tool_name=self.tool_name,
            )
            response_json.update(file_info)
            return response_json
        file = await parse_file_from_response(
            response, file_manager, file_infos=returns_file_infos, file_metadata=file_metadata
        )

        if file is not None:
            if len(returns_file_infos) == 0:
                return {self.tool_view.returns_ref_uri: file.id}

            file_name = list(returns_file_infos.keys())[0]
            return {file_name: file.id}

        if len(returns_file_infos) == 0:
            return response.json()

        raise RemoteToolError(
            f"<{list(returns_file_infos.keys())}> are defined but cannot be processed from the "
            "response. Please ensure that the response headers contain either the Content-Disposition "
            "or Content-Type field.",
            stage="Output parsing",
        )

    def function_call_schema(self) -> dict:
        schema = self.tool_view.function_call_schema()

        if len(self.examples) > 0:
            schema["examples"] = [example.to_dict() for example in self.examples]

        return schema or {}

    def __adhoc_post_process__(self, tool_response: dict) -> dict:
        # temporary adhoc post processing logic for certain toolkits
        if self.tool_name.startswith("official-doc-rec") and self.tool_name.endswith("office_doc_rec"):
            if "results" in tool_response and isinstance(tool_response["results"], list):
                reformatted_result = []
                for result_line in tool_response["results"]:
                    if "words" in result_line and "word" in result_line["words"]:
                        reformatted_result.append(result_line["words"]["word"])
                tool_response["results"] = reformatted_result
        elif self.tool_name.startswith("highacc-ocr") and self.tool_name.endswith("OCR"):
            if "words_result" in tool_response and isinstance(tool_response["words_result"], list):
                reformatted_result = []
                for result in tool_response["words_result"]:
                    if "words" in result:
                        reformatted_result.append(result["words"])
                tool_response["words_result"] = reformatted_result
        elif self.tool_name.startswith("doc-analysis") and self.tool_name.endswith("doc_analysis"):
            if "results" in tool_response and isinstance(tool_response["results"], list):
                reformatted_result = []
                for result in tool_response["results"]:
                    if "words" in result and "word" in result["words"]:
                        reformatted_result.append(result["words"]["word"])
                tool_response["results"] = reformatted_result
        elif self.tool_name.startswith("shopping-receipt") and self.tool_name.endswith("shopping_receip"):
            if "words_result" in tool_response and isinstance(tool_response["words_result"], list):
                keys = [
                    "shop_name",
                    "receipt_num",
                    "machine_num",
                    "employee_num",
                    "consumption_date",
                    "consumption_time",
                    "total_amount",
                    "change",
                    "currency",
                    "paid_amount",
                    "discount",
                    "print_time",
                ]
                for result in tool_response["words_result"]:
                    for key in keys:
                        if (
                            key in result
                            and len(result[key]) > 0
                            and "word" in result[key][0]
                            and result[key][0]["word"] == ""
                        ):
                            result.pop(key)
        return tool_response

    async def _get_file_manager(self) -> FileManager:
        if self.file_manager is None:
            file_manager = await GlobalFileManagerHandler().get()
        else:
            file_manager = self.file_manager
        return file_manager


class RemoteToolRegistor:
    def __init__(self) -> None:
        self.tool_map: Dict[str, Type[RemoteTool]] = {}

    _instance: Optional[RemoteToolRegistor] = None

    def __call__(self, name: str):
        def inner_decorator(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            RemoteToolRegistor.instance().add_tool_map(name, func)
            return wrapper

        return inner_decorator

    def add_tool_map(self, name: str, tool_class: Type[RemoteTool]):
        self.tool_map[name] = tool_class

    def get_tool_class(self, name: str) -> Type[RemoteTool]:
        if name in self.tool_map:
            return self.tool_map[name]
        return RemoteTool

    @staticmethod
    def instance() -> RemoteToolRegistor:
        if RemoteToolRegistor._instance is None:
            RemoteToolRegistor._instance = RemoteToolRegistor()
        return RemoteToolRegistor._instance


tool_registor = RemoteToolRegistor.instance()
