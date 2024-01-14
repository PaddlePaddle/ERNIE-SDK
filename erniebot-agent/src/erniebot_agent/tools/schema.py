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

import inspect
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union, get_args

from pydantic import BaseModel, ConfigDict, Field, create_model
from pydantic.fields import FieldInfo

from erniebot_agent.utils.common import create_enum_class
from erniebot_agent.utils.exceptions import RemoteToolError, ToolError

INVALID_FIELD_NAME = "__invalid_field_name__"

_logger = logging.getLogger(__name__)


def is_optional_type(type: Optional[Type]):
    args = get_args(type)
    if len(args) == 0:
        return False

    return len([arg for arg in args if arg is None.__class__]) > 0


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
        bytes: "string",
        float: "number",
        ToolParameterView: "object",
    }

    if inspect.isclass(type) and issubclass(type, ToolParameterView):
        return "object"

    if getattr(type, "_name", None) == "List":
        return "array"

    if type not in mapping:
        args = [arg for arg in get_args(type) if arg is not None.__class__]
        if len(args) > 1 or len(args) == 0:
            raise ValueError(
                "only support simple type: FieldType=int/str/float/ToolParameterView, "
                "so the target type should be one of: FieldType, List[FieldType], "
                f"Optional[FieldType], but receive {type}"
            )
        type = args[0]

    if type in mapping:
        return mapping[type]

    if inspect.isclass(type) and issubclass(type, ToolParameterView):
        return "object"

    return str(type)


def python_type_from_json_type(json_type_dict: dict) -> Type[object]:
    simple_types = {"integer": int, "string": str, "number": float, "object": ToolParameterView}
    format = json_type_dict.get("format", None)

    if json_type_dict["type"] == "string" and format == "binary":
        return bytes

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
    json_schema_extra: Optional[Dict[str, str]] = None
    description: Optional[str] = None
    required: Optional[List[str]] = None
    enum: Optional[List[Union[int, str]]] = None
    items: dict = Field(default_factory=dict)
    properties: dict = Field(default_factory=dict)


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
    elif field_info.annotation is not None and is_optional_type(field_info.annotation):
        field_type = json_type(get_args(field_info.annotation)[0])
    elif field_info.annotation is not None and issubclass(field_info.annotation, Enum):
        field_type = json_type(type(list(field_info.annotation.__members__.keys())[0]))
    else:
        field_type = json_type(field_info.annotation)

    property: Dict[str, Any] = {
        "type": field_type,
        "description": field_info.description,
    }

    if property["type"] == "array":
        if typing_list_type == "object":
            list_type: Type[ToolParameterView] = get_args(field_info.annotation)[0]
            property["items"] = list_type.to_openapi_dict()
        else:
            if not isinstance(field_info.json_schema_extra, dict):
                raise RemoteToolError("<field_info.json_schema_extra> must be dict data", stage="Loading")

            if "array_items_schema" in field_info.json_schema_extra:
                items_schema: Any = field_info.json_schema_extra["array_items_schema"]

                if not isinstance(items_schema, dict):
                    raise RemoteToolError("<array_items_schema> must be dict data", stage="Loading")

                property["items"] = {
                    "type": items_schema["type"],
                }
                if "description" in items_schema:
                    property["items"]["description"] = items_schema["description"]
            else:
                property["items"] = {"type": typing_list_type}

    elif property["type"] == "object":
        if is_optional_type(field_info.annotation):
            field_type_class: Any = get_args(field_info.annotation)[0]
        else:
            field_type_class = field_info.annotation

        openapi_dict = field_type_class.to_openapi_dict()
        property.update(openapi_dict)
    elif field_info.annotation is not None and issubclass(field_info.annotation, Enum):
        property["enum"] = list(field_info.annotation.__members__.keys())

    property["description"] = property.get("description", "")
    return OpenAPIProperty(**property)


class ToolParameterView(BaseModel):
    __prompt__: Optional[str] = None
    model_config = ConfigDict(use_enum_values=True)

    @classmethod
    def from_openapi_dict(cls, schema: dict) -> Type[ToolParameterView]:
        """parse openapi component schemas to ParameterView
        Args:
            response_or_returns (dict): the content of status code

        Returns:
            _type_: _description_
        """

        # TODO(wj-Mcat): to load Optional field
        fields = {}
        for field_name, field_dict in schema.get("properties", {}).items():
            # skip loading invalid field to improve compatibility
            if "type" not in field_dict:
                raise ToolError(f"`type` field not found in `{field_name}` property", stage="Loading")

            if "description" not in field_dict:
                raise ToolError(f"`description` field not found in `{field_name}` property", stage="Loading")

            if field_name.startswith("__"):
                continue

            field_type = python_type_from_json_type(field_dict)

            if field_type is List[ToolParameterView]:
                SubParameterView: Type[ToolParameterView] = ToolParameterView.from_openapi_dict(
                    field_dict["items"]
                )
                field_type = List[SubParameterView]  # type: ignore
            elif field_type is ToolParameterView:
                field_type = ToolParameterView.from_openapi_dict(field_dict)
            elif "enum" in field_dict:
                field_type = create_enum_class(field_name, field_dict["enum"])

            # TODO(wj-Mcat): remove supporting for `summary` field
            if "summary" in field_dict:
                description = field_dict["summary"]
                _logger.info("`summary` field will be deprecated, please use `description`")

                if "description" in field_dict:
                    _logger.info("`description` field will be used instead of `summary`")
                    description = field_dict["description"]
            else:
                description = field_dict.get("description", None)

            description = description or ""

            format = field_dict.get("format", None)
            json_schema_extra = {}
            if format is not None:
                json_schema_extra["format"] = format

            json_schema_extra.update(
                {key: value for key, value in field_dict.items() if key.startswith("x-ebagent")}
            )

            if get_typing_list_type(field_type) is not None and field_type is not List[ToolParameterView]:
                json_schema_extra["array_items_schema"] = field_dict["items"]

            field_info_param = dict(
                annotation=field_type, description=description, json_schema_extra=json_schema_extra
            )
            if "default" in field_dict:
                field_info_param["default"] = field_dict["default"]
            field = FieldInfo(**field_info_param)  # type: ignore

            # TODO(wj-Mcat): to handle list field required & not-required
            # if get_typing_list_type(field_type) is not None:
            #     field.default_factory = list

            fields[field_name] = (field_type, field)

        model = create_model("OpenAPIParameterView", __base__=ToolParameterView, **fields)  # type: ignore

        # get the prompt for schema
        model.__prompt__ = schema.get("x-ebagent-prompt", None)
        return model

    @classmethod
    def to_openapi_dict(cls) -> dict:
        """convert ParametersView to openapi spec dict

        Returns:
            dict: schema of openapi
        """

        required_names, properties = [], {}
        for field_name, field_info in cls.model_fields.items():
            if field_info.is_required() and not is_optional_type(field_info.annotation):
                required_names.append(field_name)

            properties[field_name] = dict(get_field_openapi_property(field_info))

        result = {
            "type": "object",
            "properties": properties,
        }
        if len(required_names) > 0:
            result["required"] = required_names
        result = scrub_dict(result, remove_empty_dict=True)  # type: ignore
        return result or {}

    @classmethod
    def function_call_schema(cls) -> dict:
        """get function_call schame

        Returns:
            dict: the schema of function_call
        """
        return cls.to_openapi_dict()

    @classmethod
    def from_dict(cls, field_map: Dict[str, Any]):
        """
        Class method to create a Pydantic model dynamically based on a dictionary.

        Args:
            field_map (Dict[str, Any]): A dictionary mapping field names to their corresponding type
            and description.

        Returns:
            PydanticModel: A dynamically created Pydantic model with fields specified by the
            input dictionary.

        Note:
            This method is used to create a Pydantic model dynamically based on the provided dictionary,
            where each field's type and description are specified in the input.

        """
        fields = {}
        for field_name, field_dict in field_map.items():
            field_type = field_dict["type"]
            description = field_dict["description"]
            field = FieldInfo(annotation=field_type, description=description)
            fields[field_name] = (field_type, field)
        return create_model(cls.__name__, __base__=ToolParameterView, **fields)  # type: ignore


@dataclass
class RemoteToolView:
    uri: str
    method: str
    name: str
    description: str
    version: str

    parameters: Optional[Type[ToolParameterView]] = None
    parameters_description: Optional[str] = None
    parameters_content_type: Optional[str] = None

    returns: Optional[Type[ToolParameterView]] = None
    returns_description: Optional[str] = None
    returns_content_type: Optional[str] = None

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
                        self.returns_content_type: {
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
                    self.parameters_content_type: {
                        "schema": {"$ref": "#/components/schemas/" + (self.parameters_ref_uri or "")}
                    }
                },
            }
            result["requestBody"] = parameters
        return {self.method: result}

    @staticmethod
    def from_openapi_dict(
        uri: str,
        method: str,
        path_info: dict,
        parameters_views: dict[str, Type[ToolParameterView]],
        version: str,
    ) -> RemoteToolView:
        """construct RemoteToolView from openapi spec-dict info

        Args:
            uri (str): the url path of remote tool
            method (str): http method: one of [get, post, put, delete]
            path_info (dict): the spec info of remote tool
            parameters_views (dict[str, ParametersView]):
                the dict of parameters views which are the schema of input/output of tool
            version (Optional[str]): the optional version of remote tool

        Returns:
            RemoteToolView: the instance of remote tool view
        """
        parameters_ref_uri, returns_ref_uri = None, None
        parameters, parameters_description = None, None
        parameters_content_type, returns_content_type = "application/json", None
        if "requestBody" in path_info:
            request_content = path_info["requestBody"]["content"]
            assert len(request_content.keys()) == 1
            parameters_content_type = list(request_content.keys())[0]
            request_ref = request_content[parameters_content_type]["schema"]["$ref"]
            parameters_ref_uri = request_ref.split("/")[-1]
            assert parameters_ref_uri in parameters_views
            parameters = parameters_views[parameters_ref_uri]
            parameters_description = path_info["requestBody"].get("description", None)

        returns, returns_description = None, None
        if "responses" in path_info:
            response_content = list(path_info["responses"].values())[0]["content"]
            assert len(response_content.keys()) == 1
            returns_content_type = list(response_content.keys())[0]

            # support ref in components.schemas
            if "$ref" in response_content[returns_content_type]["schema"]:
                response_ref = response_content[returns_content_type]["schema"]["$ref"]
                returns_ref_uri = response_ref.split("/")[-1]
                assert returns_ref_uri in parameters_views
                returns = parameters_views[returns_ref_uri]
                returns_description = list(path_info["responses"].values())[0].get("description", None)

        return RemoteToolView(
            name=path_info.get("operationId", uri.strip("/").replace("/", "-")),
            parameters=parameters,
            version=version,
            parameters_description=parameters_description,
            parameters_content_type=parameters_content_type,
            returns=returns,
            returns_description=returns_description,
            returns_content_type=returns_content_type,
            description=path_info.get("description", path_info.get("summary", None)),
            method=method,
            uri=uri,
            # save ref id info
            returns_ref_uri=returns_ref_uri,
            parameters_ref_uri=parameters_ref_uri,
        )

    def function_call_schema(self):
        inputs = {
            "name": self.name,
            "description": self.description,
        }
        if self.parameters is not None:
            inputs["parameters"] = self.parameters.function_call_schema()  # type: ignore
        else:
            inputs["parameters"] = {"type": "object", "properties": {}}

        if self.returns is not None:
            inputs["responses"] = self.returns.function_call_schema()  # type: ignore
        return scrub_dict(inputs) or {}


@dataclass
class Endpoint:
    url: str
    description: Optional[str] = None


@dataclass
class EndpointInfo:
    title: str
    version: str

    description: Optional[str] = None
