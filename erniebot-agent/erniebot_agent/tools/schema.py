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

from dataclasses import asdict, dataclass
from typing import List, Optional, Type, get_args

from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo
from yaml import safe_dump

INVALID_FIELD_NAME = "__invalid_field_name__"


def get_typing_list_type(type):
    """get typing.List[T] element type

    Args:
        type (typing.List): Generics type
    """
    # 1. checking list type
    if getattr(type, "_name", None) != "List":
        return None

    arg_type = get_args(type)[0]
    return json_type(arg_type)


def json_type(type: Optional[Type[object]] = None):
    if type is None:
        return "object"

    mapping = {
        int: "integer",
        str: "string",
        list: "array",
        List: "array",
        float: "number",
        ToolParameterView: "object",
    }
    if type in mapping:
        return mapping[type]

    # List[int], List[str], List[...]
    if getattr(type, "_name", None) == "List":
        return "array"

    if issubclass(type, ToolParameterView):
        return "object"

    return str(type)


def python_type_from_json_type(json_type_dict: dict) -> Type[object]:
    simple_types = {"integer": int, "string": str, "number": float, "object": ToolParameterView}
    if json_type_dict["type"] in simple_types:
        return simple_types[json_type_dict["type"]]

    assert (
        json_type_dict["type"] == "array"
    ), f"only support simple_types<{','.join(simple_types)}> and array type"
    assert "type" in json_type_dict["items"], "<items> field must be defined when 'type'=array"

    json_type_value = json_type_dict["items"]["type"]
    if json_type_value == "string":
        return List[str]
    if json_type_value == "integer":
        return List[int]
    if json_type_value == "number":
        return List[float]
    if json_type_value == "object":
        return List[ToolParameterView]

    raise ValueError(f"unsupported data type: {json_type_value}")


def scrub_dict(d: dict, remove_empty_dict: bool = False) -> Optional[dict]:
    """remove empty Value node,

        function_call_schema: require

    Args:
        d (dict): the instance of dictionary
        remove_empty_dict (bool): whether remove empty dict

    Returns:
        dict: the dictionary data after slimming down
    """
    if type(d) is dict:
        result = {}
        for k, v in d.items():
            v = scrub_dict(v, remove_empty_dict)
            if v is not None:
                result[k] = v

        if len(result) == 0:
            if not remove_empty_dict:
                return {}
            return None

        return result
    elif isinstance(d, list):
        return [scrub_dict(item, remove_empty_dict) for item in d]  # type: ignore
    else:
        return d


class OpenAPIProperty(BaseModel):
    type: str
    description: str
    items: dict = Field(default_factory=dict)


def get_field_openapi_property(field_info: FieldInfo) -> OpenAPIProperty:
    """convert pydantic FieldInfo instance to OpenAPIProperty value

    Args:
        field_info (FieldInfo): the field instance

    Returns:
        OpenAPIProperty: the converted OpenAPI Property
    """
    typing_list_type = get_typing_list_type(field_info.annotation)
    if typing_list_type is not None:
        field_type = "array"
    else:
        field_type = json_type(field_info.annotation)

    property = {
        "type": field_type,
        "description": field_info.description,
    }

    if property["type"] == "array":
        if typing_list_type == "object":
            property["items"] = {"type": field_info.annotation.to_openapi_dict()}
        else:
            property["items"] = {"type": typing_list_type}

    return OpenAPIProperty(**property)


class ToolParameterView(BaseModel):
    @classmethod
    def from_openapi_dict(cls, name, schema: dict) -> Type[ToolParameterView]:
        """parse openapi component schemas to ParameterView
        Args:
            response_or_returns (dict): the content of status code

        Returns:
            _type_: _description_
        """

        # TODO(wj-Mcat): to load Optional field
        fields = {}
        for field_name, field_dict in schema.get("properties", {}).items():
            field_type = python_type_from_json_type(field_dict)

            if field_type is List[ToolParameterView]:
                SubParameterView: Type[ToolParameterView] = ToolParameterView.from_openapi_dict(
                    field_name, field_dict["items"]
                )
                field_type = List[SubParameterView]  # type: ignore

            field = FieldInfo(annotation=field_type, description=field_dict["description"])

            # TODO(wj-Mcat): to handle list field required & not-required
            # if get_typing_list_type(field_type) is not None:
            #     field.default_factory = list

            fields[field_name] = (field_type, field)

        return create_model("OpenAPIParameterView", __base__=ToolParameterView, **fields)

    @classmethod
    def to_openapi_dict(cls) -> dict:
        """convert ParametersView to openapi spec dict

        Returns:
            dict: schema of openapi
        """
        required_names, properties = [], {}
        for field_name, field_info in cls.model_fields.items():
            if field_info.is_required():
                required_names.append(field_name)

            properties[field_name] = dict(get_field_openapi_property(field_info))

        result = {
            "type": "object",
            "required": required_names,
            "properties": properties,
        }
        result = scrub_dict(result, remove_empty_dict=True)  # type: ignore
        return result or {}

    @classmethod
    def function_call_schema(cls) -> dict:
        """get function_call schame

        Returns:
            dict: the schema of function_call
        """
        return cls.to_openapi_dict()


@dataclass
class RemoteToolView:
    uri: str
    method: str
    name: str
    description: str
    parameters: Optional[Type[ToolParameterView]] = None
    parameters_description: Optional[str] = None
    returns: Optional[Type[ToolParameterView]] = None
    returns_description: Optional[str] = None

    returns_ref_uri: Optional[str] = None
    parameters_ref_uri: Optional[str] = None

    def to_openapi_dict(self):
        result = {
            "operationId": self.name,
            "description": self.description,
        }
        if self.returns is not None:
            response = {
                "200": {
                    "description": self.returns_description,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/" + (self.returns_ref_uri or "")}
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
                        "schema": {"$ref": "#/components/schemas/" + (self.parameters_ref_uri or "")}
                    }
                },
            }
            result["requestBody"] = parameters
        return {self.method: result}

    @staticmethod
    def from_openapi_dict(
        uri: str, method: str, path_info: dict, parameters_views: dict[str, Type[ToolParameterView]]
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
        parameters_ref_uri, returns_ref_uri = None, None
        parameters, parameters_description = None, None
        if "requestBody" in path_info:
            request_ref = path_info["requestBody"]["content"]["application/json"]["schema"]["$ref"]
            parameters_ref_uri = request_ref.split("/")[-1]
            assert parameters_ref_uri in parameters_views
            parameters = parameters_views[parameters_ref_uri]
            parameters_description = path_info["requestBody"].get("description", None)

        returns, returns_description = None, None
        if "responses" in path_info:
            response_ref = list(path_info["responses"].values())[0]["content"]["application/json"]["schema"][
                "$ref"
            ]
            returns_ref_uri = response_ref.split("/")[-1]
            assert returns_ref_uri in parameters_views
            returns = parameters_views[returns_ref_uri]
            returns_description = list(path_info["responses"].values())[0].get("description", None)

        return RemoteToolView(
            name=path_info["operationId"],
            parameters=parameters,
            parameters_description=parameters_description,
            returns=returns,
            returns_description=returns_description,
            description=path_info.get("description", path_info.get("summary", None)),
            method=method,
            uri=uri,
            # save ref id info
            returns_ref_uri=returns_ref_uri,
            parameters_ref_uri=parameters_ref_uri,
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

    component_schemas: dict[str, Type[ToolParameterView]]

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

    @staticmethod
    def from_openapi_file(file: str) -> PluginSchema:
        """only support openapi v3.0.1

        Args:
            file (str): the path of openapi yaml file
        """
        from openapi_spec_validator import validate
        from openapi_spec_validator.readers import read_from_filename

        spec_dict, _ = read_from_filename(file)
        validate(spec_dict)

        # info
        info = EndpointInfo(**spec_dict["info"])
        servers = [Endpoint(**server) for server in spec_dict.get("servers", [])]

        # components
        component_schemas = spec_dict["components"]["schemas"]
        fields = {}
        for name, schema in component_schemas.items():
            parameter_view = ToolParameterView.from_openapi_dict(name, schema)
            fields[name] = parameter_view

        # paths
        paths = []
        for path, path_info in spec_dict.get("paths", {}).items():
            for method, path_method_info in path_info.items():
                paths.append(
                    RemoteToolView.from_openapi_dict(
                        uri=path,
                        method=method,
                        path_info=path_method_info,
                        parameters_views=fields,
                    )
                )

        return PluginSchema(
            openapi=spec_dict["openapi"],
            info=info,
            servers=servers,
            paths=paths,
            component_schemas=fields,
        )
