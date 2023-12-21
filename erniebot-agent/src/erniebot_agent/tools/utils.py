import base64
import inspect
import re
import typing
from copy import deepcopy
from typing import Any, Dict, Optional, Type

from openapi_spec_validator import validate
from openapi_spec_validator.readers import read_from_filename
from requests import Response

from erniebot_agent.file_io.base import File
from erniebot_agent.file_io.file_manager import FileManager
from erniebot_agent.file_io.protocol import is_local_file_id, is_remote_file_id
from erniebot_agent.tools.schema import (
    ToolParameterView,
    get_args,
    get_typing_list_type,
)
from erniebot_agent.utils.common import get_file_suffix, is_json_response
from erniebot_agent.utils.exception import RemoteToolError
from erniebot_agent.utils.logging import logger


def tool_response_contains_file(element: Any):
    if isinstance(element, str):
        if is_local_file_id(element) or is_remote_file_id(element):
            return True
    elif isinstance(element, dict):
        for val in element.values():
            if tool_response_contains_file(val):
                return True
    elif isinstance(element, list):
        for val in element:
            if tool_response_contains_file(val):
                return True


def validate_openapi_yaml(yaml_file: str) -> bool:
    """do validation on the yaml file

    Args:
        yaml_file (str): the path of yaml file

    Returns:
        bool: whether yaml file is valid
    """
    yaml_dict = read_from_filename(yaml_file)[0]
    try:
        validate(yaml_dict)
        return True
    except Exception as e:  # type: ignore
        logger.error(e)
        return False


def get_file_info_from_param_view(
    param_view: Optional[Type[ToolParameterView]] = None,
) -> Dict[str, Dict[str, str]]:
    """get file names from tool parameter view

    Args:
        param_view (ToolParameterView): the ToolParameterView pydantic class

    Returns:
        List[str]: the names of file
    """
    if param_view is None:
        return {}

    file_infos: Dict[str, Any] = {}
    for key in param_view.model_fields.keys():
        model_field = param_view.model_fields[key]

        list_base_annotation = get_typing_list_type(model_field.annotation)
        json_schema_extra: dict = typing.cast(dict, model_field.json_schema_extra)

        if list_base_annotation is not None:
            if list_base_annotation == "object":
                # get base type
                arg_type = get_args(model_field.annotation)[0]
                sub_file_infos = get_file_info_from_param_view(arg_type)
                if len(sub_file_infos) > 0:
                    file_infos[key] = sub_file_infos
                continue

            if "array_items_schema" in json_schema_extra:
                json_schema_extra = json_schema_extra["array_items_schema"]
                if not isinstance(json_schema_extra, dict):
                    raise RemoteToolError(
                        f"<array_items_schema> field must be dict type in model_field<{key}> "
                        f"with the field_info<{model_field}>. Please check the format of yaml "
                        "in current tool.",
                        stage="Output parsing",
                    )

        elif model_field.annotation is not None and issubclass(model_field.annotation, ToolParameterView):
            sub_file_infos = get_file_info_from_param_view(model_field.annotation)
            if len(sub_file_infos) > 0:
                file_infos[key] = sub_file_infos
            continue

        if json_schema_extra and json_schema_extra.get("format", None) in [  # type: ignore
            "byte",
            "binary",
        ]:
            file_infos[key] = deepcopy(json_schema_extra)
    return file_infos


async def parse_file_from_json_response(
    json_data: dict, file_manager: FileManager, param_view: Type[ToolParameterView], tool_name: str
):
    if param_view is None:
        return {}

    file_infos: Dict[str, Any] = {}
    for key in param_view.model_fields.keys():
        model_field = param_view.model_fields[key]

        # to avoid: yaml schema is not matched with json response schema
        if key not in json_data:
            continue

        list_base_annotation = get_typing_list_type(model_field.annotation)
        if list_base_annotation == "object":
            # get base type
            arg_type = get_args(model_field.annotation)[0]
            sub_file_infos_list = []
            for json_item in json_data[key]:
                sub_file_infos = await parse_file_from_json_response(
                    json_item, file_manager=file_manager, param_view=arg_type, tool_name=tool_name
                )
                if sub_file_infos:
                    sub_file_infos_list.append(sub_file_infos)
            if sub_file_infos_list:
                file_infos[key] = sub_file_infos_list
        elif inspect.isclass(model_field.annotation) and issubclass(
            model_field.annotation, ToolParameterView
        ):
            sub_file_infos = await parse_file_from_json_response(
                json_data[key],
                file_manager=file_manager,
                param_view=model_field.annotation,
                tool_name=tool_name,
            )
            if sub_file_infos:
                file_infos[key] = sub_file_infos
        else:
            json_schema_extra = model_field.json_schema_extra
            if not isinstance(json_schema_extra, dict):
                raise RemoteToolError(
                    f"<json_schema_extra> field must be dict type in model_field<{key}> "
                    f"with the field_info<{model_field}>. Please check the format of yaml in current tool.",
                    stage="Output parsing",
                )

            format = json_schema_extra.get("format", None)
            if format in ["byte", "binary"]:
                content = json_data[key]
                if format == "byte":
                    content = base64.b64decode(content)

                mime_type = json_schema_extra.get("x-ebagent-file-mime-type", None)
                if mime_type is not None and not isinstance(mime_type, str):
                    raise RemoteToolError(
                        f"x-ebagent-file-mime-type value must be None or string in key<{key}>, "
                        f"but receive ({type(mime_type)})<{mime_type}>",
                        stage="Output parsing",
                    )

                suffix = get_file_suffix(mime_type)
                file = await file_manager.create_file_from_bytes(
                    content,
                    filename=f"test{suffix}",
                    file_purpose="assistants_output",
                    file_metadata={"tool_name": tool_name},
                )
                file_infos[key] = file.id
    return file_infos


async def parse_file_from_response(
    response: Response,
    file_manager: FileManager,
    file_infos: Dict[str, Dict[str, str]],
    file_metadata: Dict[str, str],
) -> Optional[File]:
    # 1. parse file by `Content-Disposition`
    content_disposition = response.headers.get("Content-Disposition", None)
    if content_disposition is not None:
        file_name = response.headers["Content-Disposition"].split("filename=")[1]
        local_file = await file_manager.create_file_from_bytes(
            response.content, file_name, file_purpose="assistants_output", file_metadata=file_metadata
        )
        return local_file

    # 2. parse file from file_mimetypes
    if len(file_infos) > 1:
        raise RemoteToolError(
            "Multiple file MIME types are defined in the Response Schema. Currently, only single "
            "file output is supported. Please ensure that only one file MIME type is defined in "
            "the Response Schema.",
            stage="Output parsing",
        )

    if len(file_infos) == 1:
        file_name = list(file_infos.keys())[0]
        file_mimetype = file_infos[file_name].get("x-ebagent-file-mime-type", None)
        if file_mimetype is not None:
            file_suffix = get_file_suffix(file_mimetype)
            content = response.content
            if file_infos[file_name].get("format", None) == "byte":
                content = base64.b64decode(content)

            return await file_manager.create_file_from_bytes(
                content,
                f"tool-{file_suffix}",
                file_purpose="assistants_output",
                file_metadata=file_metadata,
            )

    # 3. parse file by content_type
    content_type = response.headers.get("Content-Type", None)
    if content_type is not None:
        file_suffix = get_file_suffix(content_type)
        return await file_manager.create_file_from_bytes(
            response.content,
            f"tool-{file_suffix}",
            file_purpose="assistants_output",
            file_metadata=file_metadata,
        )

    if is_json_response(response):
        raise RemoteToolError(
            "Can not parse file from response: the type of data from response is json",
            stage="Output parsing",
        )
    raise RemoteToolError(
        "Can not parse file from response: the type of data from response is not json, "
        "and can not find `Content-Disposition` or `Content-Type` field from response header.",
        stage="Output parsing",
    )


def is_base64_string(element: Any) -> bool:
    """check whether a string is base64 sdtring

    refer to: https://stackoverflow.com/a/8571649

    Args:
        element (str): the content of string

    Returns:
        bool: whether is base64 string
    """
    if not isinstance(element, str):
        return False

    if len(element) < 100:
        return False

    expression = "^([A-Za-z0-9+/]{4})*([A-Za-z0-9+/]{3}=|[A-Za-z0-9+/]{2}==)?$"
    matches = re.match(expression, element)
    return matches is not None
