from __future__ import annotations

import copy
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Type

import erniebot
import requests
from openapi_spec_validator.readers import read_from_filename
from yaml import safe_dump

from erniebot_agent.file_io import get_file_manager
from erniebot_agent.file_io.file_manager import FileManager
from erniebot_agent.messages import AIMessage, FunctionCall, HumanMessage, Message
from erniebot_agent.tools.remote_tool import RemoteTool, tool_registor
from erniebot_agent.tools.schema import (
    Endpoint,
    EndpointInfo,
    RemoteToolView,
    ToolParameterView,
    scrub_dict,
)
from erniebot_agent.tools.utils import validate_openapi_yaml
from erniebot_agent.utils.exception import RemoteToolError
from erniebot_agent.utils.http import url_file_exists
from erniebot_agent.utils.logging import logger


@dataclass
class RemoteToolkit:
    """RemoteToolkit can be converted by openapi.yaml and endpoint"""

    openapi: str
    info: EndpointInfo
    servers: List[Endpoint]
    paths: List[RemoteToolView]
    file_manager: FileManager

    component_schemas: dict[str, Type[ToolParameterView]]
    headers: dict
    examples: List[Message] = field(default_factory=list)
    _AISTUDIO_HUB_BASE_URL: ClassVar[str] = "https://aistudio-hub.baidu.com"

    @property
    def tool_name_prefix(self) -> str:
        return f"{self.info.title}/{self.info.version}"

    def __getitem__(self, tool_name: str) -> RemoteTool:
        return self.get_tool(tool_name)

    def get_tools(self) -> List[RemoteTool]:
        TOOL_CLASS = tool_registor.get_tool_class(self.info.title)
        return [
            TOOL_CLASS(
                copy.deepcopy(path),
                self.servers[0].url,
                self.headers,
                self.info.version,
                file_manager=self.file_manager,
                examples=self.get_examples_by_name(path.name),
                tool_name_prefix=self.tool_name_prefix,
            )
            for path in self.paths
        ]

    def get_examples_by_name(self, tool_name: str) -> List[Message]:
        """get examples by tool-name

        Args:
            tool_name (str): the name of the tool

        Returns:
            List[Message]: the messages
        """
        # 1. split messages
        tool_examples: List[List[Message]] = []
        examples: List[Message] = []
        for example in self.examples:
            if isinstance(example, HumanMessage):
                if len(examples) == 0:
                    examples.append(example)
                else:
                    tool_examples.append(examples)
                    examples = [example]
            else:
                examples.append(example)

        if len(examples) > 0:
            tool_examples.append(examples)

        final_exampels: List[Message] = []
        # 2. find the target tool examples or empty messages
        for examples in tool_examples:
            tool_names = [
                example.function_call.get("name", None)
                for example in examples
                if isinstance(example, AIMessage) and example.function_call is not None
            ]
            tool_names = [name for name in tool_names if name]

            if tool_name in tool_names:
                # 3. prepend `tool_name_prefix` to all tool names in examples
                for example in examples:
                    if isinstance(example, AIMessage) and example.function_call is not None:
                        original_tool_name = example.function_call["name"]
                        example.function_call["name"] = f"{self.tool_name_prefix}/{original_tool_name}"
                final_exampels.extend(examples)

        return final_exampels

    def get_tool(self, tool_name: str) -> RemoteTool:
        paths = [path for path in self.paths if path.name == tool_name]
        if len(paths) == 0:
            raise RemoteToolError(
                f"`{tool_name}` not found under RemoteToolkit `{self.tool_name_prefix}`", stage="Loading"
            )
        elif len(paths) > 1:
            raise RemoteToolError(
                f"Found duplicate `{tool_name}` under RemoteToolkit `{self.tool_name_prefix}`",
                stage="Loading",
            )

        TOOL_CLASS = tool_registor.get_tool_class(self.info.title)
        return TOOL_CLASS(
            paths[0],
            self.servers[0].url,
            self.headers,
            self.info.version,
            file_manager=self.file_manager,
            examples=self.get_examples_by_name(tool_name),
            tool_name_prefix=self.tool_name_prefix,
        )

    def to_openapi_dict(self) -> dict:
        """convert plugin schema to openapi spec dict"""
        spec_dict = {
            "openapi": self.openapi,
            "info": asdict(self.info),
            "servers": [asdict(server) for server in self.servers],
            "paths": {tool_view.uri: tool_view.to_openapi_dict() for tool_view in self.paths},
            "components": {
                "schemas": {
                    uri: parameters_view.to_openapi_dict()
                    for uri, parameters_view in self.component_schemas.items()
                }
            },
        }
        return scrub_dict(spec_dict, remove_empty_dict=True) or {}

    def to_openapi_file(self, file: str):
        """generate openapi configuration file

        Args:
            file (str): the path of the openapi yaml file
        """
        spec_dict = self.to_openapi_dict()
        with open(file, "w+", encoding="utf-8") as f:
            safe_dump(spec_dict, f, indent=4)

    @classmethod
    def from_openapi_dict(
        cls,
        openapi_dict: Dict[str, Any],
        access_token: Optional[str] = None,
        file_manager: Optional[FileManager] = None,
    ) -> RemoteToolkit:
        info = EndpointInfo(**openapi_dict["info"])
        servers = [Endpoint(**server) for server in openapi_dict.get("servers", [])]

        # components
        component_schemas = openapi_dict["components"]["schemas"]
        fields = {}
        for schema_name, schema in component_schemas.items():
            parameter_view = ToolParameterView.from_openapi_dict(schema)
            fields[schema_name] = parameter_view

        # paths
        paths = []
        for path, path_info in openapi_dict.get("paths", {}).items():
            for method, path_method_info in path_info.items():
                paths.append(
                    RemoteToolView.from_openapi_dict(
                        uri=path,
                        method=method,
                        version=info.version,
                        path_info=path_method_info,
                        parameters_views=fields,
                    )
                )

        if file_manager is None:
            file_manager = get_file_manager(access_token)

        return RemoteToolkit(
            openapi=openapi_dict["openapi"],
            info=info,
            servers=servers,
            paths=paths,
            component_schemas=fields,
            headers=cls._get_authorization_headers(access_token),
            file_manager=file_manager,
        )

    @classmethod
    def from_openapi_file(
        cls, file: str, access_token: Optional[str] = None, file_manager: Optional[FileManager] = None
    ) -> RemoteToolkit:
        """only support openapi v3.0.1

        Args:
            file (str): the path of openapi yaml file
            access_token (Optional[str]): the path of openapi yaml file
        """
        if not validate_openapi_yaml(file):
            raise RemoteToolError(f"invalid openapi yaml file: {file}", stage="Loading")

        spec_dict, _ = read_from_filename(file)
        return cls.from_openapi_dict(spec_dict, access_token=access_token, file_manager=file_manager)

    @classmethod
    def _get_authorization_headers(cls, access_token: Optional[str]) -> dict:
        if access_token is None:
            access_token = erniebot.access_token

        headers = {"Content-Type": "application/json"}
        if access_token is None:
            logger.warning("access_token is NOT provided, this may cause 403 HTTP error..")
        else:
            headers["Authorization"] = f"token {access_token}"
        return headers

    @classmethod
    def from_aistudio(
        cls,
        tool_id: str,
        version: Optional[str] = None,
        access_token: Optional[str] = None,
        file_manager: Optional[FileManager] = None,
    ) -> RemoteToolkit:
        from urllib.parse import urlparse

        aistudio_base_url = os.getenv("AISTUDIO_HUB_BASE_URL", cls._AISTUDIO_HUB_BASE_URL)
        parsed_url = urlparse(aistudio_base_url)
        tool_url = parsed_url._replace(netloc=f"tool-{tool_id}.{parsed_url.netloc}").geturl()
        return cls.from_url(tool_url, version=version, access_token=access_token, file_manager=file_manager)

    @classmethod
    def from_url(
        cls,
        url: str,
        version: Optional[str] = None,
        access_token: Optional[str] = None,
        file_manager: Optional[FileManager] = None,
    ) -> RemoteToolkit:
        # 1. download openapy.yaml file to temp directory
        if not url.endswith("/"):
            url += "/"
        openapi_yaml_url = url + ".well-known/openapi.yaml"

        if version:
            openapi_yaml_url = openapi_yaml_url + "?version=" + version

        with tempfile.TemporaryDirectory() as temp_dir:
            response = requests.get(openapi_yaml_url, headers=cls._get_authorization_headers(access_token))
            if response.status_code != 200:
                logger.debug(f"The resource requested returned the following headers: {response.headers}")
                raise RemoteToolError(
                    f"`{openapi_yaml_url}` returned {response.status_code}: {response.text}", stage="Loading"
                )

            file_content = response.content.decode("utf-8")
            if not file_content.strip():
                raise RemoteToolError(f"the content is empty from: {openapi_yaml_url}", stage="Loading")

            file_path = os.path.join(temp_dir, "openapi.yaml")
            with open(file_path, "w+", encoding="utf-8") as f:
                f.write(file_content)

            toolkit = RemoteToolkit.from_openapi_file(
                file_path, access_token=access_token, file_manager=file_manager
            )
            for server in toolkit.servers:
                server.url = url

            toolkit.examples = cls.load_remote_examples_yaml(url, access_token)

        return toolkit

    @classmethod
    def load_remote_examples_yaml(cls, url: str, access_token: Optional[str] = None) -> List[Message]:
        """load remote examples by url: url/.well-known/examples.yaml

        Args:
            url (str): the base url of the remote toolkit
        """
        if not url.endswith("/"):
            url += "/"
        examples_yaml_url = url + ".well-known/examples.yaml"
        if not url_file_exists(examples_yaml_url, cls._get_authorization_headers(access_token)):
            return []

        examples = []
        with tempfile.TemporaryDirectory() as temp_dir:
            response = requests.get(examples_yaml_url, headers=cls._get_authorization_headers(access_token))
            if response.status_code != 200:
                logger.debug(f"The resource requested returned the following headers: {response.headers}")
                raise RemoteToolError(
                    f"`{examples_yaml_url}` returned {response.status_code}: {response.text}",
                    stage="Loading",
                )

            file_content = response.content.decode("utf-8")
            if not file_content.strip():
                raise RemoteToolError(f"the content is empty from: {examples_yaml_url}", stage="Loading")

            file_path = os.path.join(temp_dir, "examples.yaml")
            with open(file_path, "w+", encoding="utf-8") as f:
                f.write(file_content)

            examples = cls.load_examples_yaml(file_path)

        return examples

    @classmethod
    def load_examples_dict(cls, examples_dict: Dict[str, Any]) -> List[Message]:
        messages: List[Message] = []
        for examples in examples_dict["examples"]:
            examples = examples["context"]
            for example in examples:
                if "user" == example["role"]:
                    messages.append(HumanMessage(example["content"]))
                elif "bot" in example["role"]:
                    plugin = example["plugin"]
                    if "operationId" in plugin:
                        function_call: FunctionCall = {
                            "name": plugin["operationId"],
                            "thoughts": plugin["thoughts"],
                            "arguments": json.dumps(plugin["requestArguments"], ensure_ascii=False),
                        }
                    else:
                        function_call = {
                            "name": "",
                            "thoughts": plugin["thoughts"],
                            "arguments": "{}",
                        }  # type: ignore
                    messages.append(AIMessage("", function_call=function_call))
                else:
                    raise RemoteToolError(f"invald role: <{example['role']}>", stage="Loading")
        return messages

    @classmethod
    def load_examples_yaml(cls, file: str) -> List[Message]:
        """load examples from yaml file

        Args:
            file (str): the path of examples file

        Returns:
            List[Message]: the list of messages
        """
        content: dict = read_from_filename(file)[0]
        if len(content) == 0 or "examples" not in content:
            raise RemoteToolError("invalid examples configuration file", stage="Loading")
        return cls.load_examples_dict(content)

    def function_call_schemas(self) -> List[dict]:
        return [tool.function_call_schema() for tool in self.get_tools()]
