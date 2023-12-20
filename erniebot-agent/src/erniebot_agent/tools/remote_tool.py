from __future__ import annotations

import base64
from copy import deepcopy
from typing import Any, Dict, List, Optional, Type

import requests

from erniebot_agent.file_io.file_manager import FileManager
from erniebot_agent.messages import Message
from erniebot_agent.tools.base import BaseTool
from erniebot_agent.tools.schema import RemoteToolView
from erniebot_agent.tools.utils import (
    get_file_info_from_param_view,
    parse_file_from_json_response,
    parse_file_from_response,
    tool_response_contains_file,
)
from erniebot_agent.utils.common import is_json_response
from erniebot_agent.utils.exception import RemoteToolError
from erniebot_agent.utils.logging import logger


class RemoteTool(BaseTool):
    def __init__(
        self,
        tool_view: RemoteToolView,
        server_url: str,
        headers: dict,
        version: str,
        file_manager: FileManager,
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
            tool_arguments[key] = tool_arguments[key].replace("<file>", "").replace("</file>", "")
            byte_str = await fileid_to_byte(tool_arguments[key], self.file_manager)
            if parameter_file_info[key]["format"] == "byte":
                byte_str = base64.b64encode(byte_str).decode()
            tool_arguments[key] = byte_str

        # 2. call tool get response
        if self.tool_view.parameters is not None:
            tool_arguments = dict(self.tool_view.parameters(**tool_arguments))

        return tool_arguments

    async def __post_process__(self, tool_response: dict) -> dict:
        if self.response_prompt is not None:
            tool_response["prompt"] = self.response_prompt
        elif self.tool_view.returns is not None and self.tool_view.returns.__prompt__ is not None:
            tool_response["prompt"] = self.tool_view.returns.__prompt__
        elif tool_response_contains_file(tool_response):
            tool_response["prompt"] = "回复中提及符合'file-'格式的字段时，请直接展示，不要将其转换为链接或添加任何HTML, Markdown等格式化元素"

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

        file_metadata = {"tool_name": self.tool_name}
        if is_json_response(response) and len(returns_file_infos) > 0:
            response_json = response.json()
            file_info = await parse_file_from_json_response(
                response_json,
                file_manager=self.file_manager,
                param_view=self.tool_view.returns,  # type: ignore
                tool_name=self.tool_name,
            )
            response_json.update(file_info)
            return response_json
        file = await parse_file_from_response(
            response, self.file_manager, file_infos=returns_file_infos, file_metadata=file_metadata
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
