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
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional

from docstring_parser import DocstringParam, DocstringReturns
from yaml import safe_dump


def scrub_dict(d):
    """remove empty Value node

    Args:
        d (dict): the instance of dictionary

    Returns:
        dict: the dictionary data after slimming down
    """
    if type(d) is dict:
        if len(d) == 0:
            return {}

        result = {}
        for k, v in d.items():
            if v is None:
                continue
            result[k] = v
        return result
    else:
        return d


@dataclass
class ParameterView:
    """function parameter object"""

    type: Optional[str] = None
    description: Optional[str] = None
    items: dict = field(default_factory=dict)
    name: Optional[str] = None
    default_value: Optional[str] = None
    required: Optional[bool] = True

    @property
    def json_type(self):
        mapping = {"int": "integer", "str": "string"}
        if self.type in mapping:
            return mapping[self.type]
        return self.type

    def to_openapi_dict(self):
        return {
            "type": self.json_type,
            "description": self.description,
            "items": self.items,
        }


@dataclass
class ParametersView:
    """a set of parameters which can be input or output of function"""

    parameters: List[ParameterView]
    name: Optional[str] = None
    code: int = 200
    type: str = "object"

    @staticmethod
    def from_docstring(params: List[DocstringParam] | DocstringReturns | None) -> Optional[ParametersView]:
        """parse docstring param to ParameterView

        Args:
            params (List[DocstringParam]): the list of parameter view

        Returns:
            ParametersView: the instance of ParametersView
        """
        if params is None:
            return None

        parameters = []
        if isinstance(params, DocstringReturns):
            # only support one return value
            return ParametersView(
                parameters=[
                    ParameterView(
                        type=params.type_name, description=params.description, name=params.return_name
                    )
                ]
            )

        for param in params:
            parameter_view = ParameterView(
                type=param.type_name or "",
                description=param.description,
                name=param.arg_name,
                default_value=param.default,
                required=not param.is_optional,
            )
            parameters.append(parameter_view)
        return ParametersView(parameters=parameters)

    @staticmethod
    def from_openapi_dict(name, schema: dict) -> ParametersView:
        """parse openapi component schemas to ParameterView
        Args:
            response_or_returns (dict): the content of status code

        Returns:
            _type_: _description_
        """
        parameters = []
        for parameter_name, parameter_info in schema.get("properties", {}).items():
            parameter = ParameterView(name=parameter_name, **parameter_info)
            parameter.required = parameter_name in schema.get("required", [])
            parameters.append(parameter)
        return ParametersView(parameters=parameters, name=name)

    def to_openapi_dict(self) -> dict:
        """convert ParametersView to openapi spec dict

        Returns:
            dict: schema of openapi
        """
        return {
            "type": "object",
            "required": [
                parameter_view.name for parameter_view in self.parameters if parameter_view.required
            ],
            "properties": {
                parameter_view.name: parameter_view.to_openapi_dict()
                for parameter_view in self.parameters
                if parameter_view.required
            },
        }

    def to_function_inputs(self) -> dict:
        if not self.parameters:
            return {"type": "object", "properties": {}}

        params = {param.name: param.to_openapi_dict() for param in self.parameters if param.name}
        return {
            "type": "object",
            "properties": params,
            "required": [param.name for param in self.parameters if param.required and param.name],
        }


@dataclass
class ToolView:
    """the view of one tool"""

    name: str
    description: str
    parameters: Optional[ParametersView] = None
    returns: Optional[ParametersView] = None
    examples: Optional[List[dict]] = None


@dataclass
class RemoteToolView:
    uri: str
    method: str
    name: str
    description: str
    parameters: Optional[ParametersView] = None
    parameters_description: Optional[str] = None
    returns: Optional[ParametersView] = None
    returns_description: Optional[str] = None

    def to_openapi_dict(self):
        result = {
            "operationId": self.name,
            "summary": self.description,
        }
        if self.returns is not None:
            response = {
                "200": {
                    "description": self.returns_description,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/" + (self.returns.name or "")}
                        }
                    },
                }
            }
            result["responses"] = response

        if self.parameters is not None:
            parameters = {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/" + (self.parameters.name or "")}
                    }
                },
            }
            result["requestBody"] = parameters
        return {self.method: result}

    @staticmethod
    def from_openapi_dict(
        uri: str, method: str, path_info: dict, parameters_views: dict[str, ParametersView]
    ) -> RemoteToolView:
        """construct RemoteToolView from openapi spec-dict info

        Args:
            uri (str): the url path of remote tool
            method (str): http method: one of [get, post, put, delete]
            path_info (dict): the spec info of remote tool
            parameters_views (dict[str, ParametersView]):
                the dict of parameters views which are the schema of input/output of tool

        Returns:
            RemoteToolView: the instance of remote tool view
        """
        parameters, parameters_description = None, None
        if "requestBody" in path_info:
            request_ref = path_info["requestBody"]["content"]["application/json"]["schema"]["$ref"]
            request_ref_uri = request_ref.split("/")[-1]
            assert request_ref_uri in parameters_views
            parameters = parameters_views[request_ref_uri]
            parameters_description = path_info["requestBody"].get("description", None)

        returns, returns_description = None, None
        if "responses" in path_info:
            response_ref = list(path_info["responses"].values())[0]["content"]["application/json"]["schema"][
                "$ref"
            ]
            response_ref_uri = response_ref.split("/")[-1]
            assert response_ref_uri in parameters_views
            returns = parameters_views[response_ref_uri]
            returns_description = list(path_info["responses"].values())[0].get("description", None)

        return RemoteToolView(
            name=path_info["operationId"],
            parameters=parameters,
            parameters_description=parameters_description,
            returns=returns,
            returns_description=returns_description,
            description=path_info["summary"],
            method=method,
            uri=uri,
        )


@dataclass
class Endpoint:
    url: str


@dataclass
class EndpointInfo:
    title: str
    description: str
    version: str


@dataclass
class PluginSchema:
    """plugin schema object which be converted from Toolkit and generate openapi configuration file"""

    openapi: str
    info: EndpointInfo
    servers: List[Endpoint]
    paths: List[RemoteToolView]

    component_schemas: dict[str, ParametersView]

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
        return scrub_dict(spec_dict)

    def to_openapi_file(self, file: str):
        """generate openapi configuration file

        Args:
            file (str): the path of the openapi yaml file
        """
        spec_dict = self.to_openapi_dict()
        with open(file, "w+", encoding="utf-8") as f:
            safe_dump(spec_dict, f, indent=4)

    @staticmethod
    def from_openapi_file(file: str) -> PluginSchema:
        """only support openapi v3.0.1

        Args:
            file (str): the path of openapi yaml file
        """
        from openapi_spec_validator import validate
        from openapi_spec_validator.readers import read_from_filename

        spec_dict, base_uri = read_from_filename(file)
        validate(spec_dict)

        # info
        info = EndpointInfo(**spec_dict["info"])
        servers = [Endpoint(**server) for server in spec_dict.get("servers", [])]

        # components
        component_schemas = spec_dict["components"]["schemas"]
        parameters_views = {}
        for name, schema in component_schemas.items():
            parameters_view = ParametersView.from_openapi_dict(name, schema)
            parameters_views[name] = parameters_view

        # paths
        paths = []
        for path, path_info in spec_dict.get("paths", {}).items():
            for method, path_method_info in path_info.items():
                paths.append(
                    RemoteToolView.from_openapi_dict(
                        uri=path,
                        method=method,
                        path_info=path_method_info,
                        parameters_views=parameters_views,
                    )
                )
        return PluginSchema(
            openapi=spec_dict["openapi"],
            info=info,
            servers=servers,
            paths=paths,
            component_schemas=parameters_views,
        )
