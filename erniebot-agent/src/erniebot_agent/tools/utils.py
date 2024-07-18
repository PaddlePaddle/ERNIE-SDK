import base64
import inspect
import logging
import typing
from copy import deepcopy
from typing import Any, Dict, Optional, Type, no_type_check

from httpx._models import Response
from openapi_spec_validator import validate
from openapi_spec_validator.readers import read_from_filename

from erniebot_agent.file import File, FileManager
from erniebot_agent.file.protocol import (
    FilePurpose,
    is_local_file_id,
    is_remote_file_id,
)
from erniebot_agent.tools.schema import (
    ToolParameterView,
    get_args,
    get_typing_list_type,
)
from erniebot_agent.utils.common import get_file_suffix, import_module, is_json_response
from erniebot_agent.utils.exceptions import RemoteToolError

_logger = logging.getLogger(__name__)


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
        _logger.error(e)
        _logger.error(
            "You can edit your openapi.yaml file in https://editor.swagger.io/ "
            "which is more friendly for you to find the issue."
        )
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


async def create_file_from_data(
    content,
    format: str,
    mime_type: str,
    file_manager: FileManager,
    file_purpose: FilePurpose = "assistants_output",
    file_metadata={},
) -> File:
    file_suffix = get_file_suffix(mime_type)
    if format == "byte":
        content = base64.b64decode(content)

    return await file_manager.create_file_from_bytes(
        content,
        f"tool-{file_suffix}",
        file_purpose=file_purpose,
        file_metadata=file_metadata,
    )


async def get_content_by_file_id(file_id: str, format: str, file_manager: FileManager) -> str:
    file_id = file_id.replace("<file>", "").replace("</file>", "")
    file = file_manager.look_up_file_by_id(file_id)
    byte_str = await file.read_contents()
    if format == "byte":
        byte_str = base64.b64encode(byte_str)

    return byte_str.decode()


def is_file_config(json_schema_extra: dict) -> bool:
    """check wheter is file-config

    Args:
        json_schema_extra (dict): the config from yaml file

    Returns:
        bool: whether is file-config
    """
    return json_schema_extra.get("format", None) in ["byte", "binary"]


@no_type_check
async def parse_json_request(
    view: Type[ToolParameterView], json_dict, file_manager: FileManager
) -> Dict[str, Any]:
    result = {}
    for field_name, model_field in view.model_fields.items():
        list_type = get_typing_list_type(model_field.annotation)

        if list_type is None and model_field.json_schema_extra is not None:
            if not isinstance(model_field.json_schema_extra, dict):
                raise RemoteToolError(
                    f"`json_schema_extra` field must be dict type, but got "
                    f"<{type(model_field.json_schema_extra)}> in model_field<{field_name}>. "
                    "Please check the format of yaml in current tool."
                )

            if model_field.annotation == str and is_file_config(model_field.json_schema_extra):
                format = model_field.json_schema_extra.get("format", None)

                if format is not None:
                    file_content = await get_content_by_file_id(
                        json_dict[field_name], format=format, file_manager=file_manager
                    )
                    result[field_name] = file_content
            elif issubclass(model_field.annotation, ToolParameterView):
                files = await parse_json_request(
                    model_field.annotation,
                    json_dict[field_name],
                    file_manager=file_manager,
                )
                if len(files) > 0:
                    result[field_name] = files

        else:
            array_json_schema = model_field.json_schema_extra.get("array_items_schema", None)
            sub_class = get_args(model_field.annotation)[0]
            if list_type == "string" and is_file_config(array_json_schema):
                format = array_json_schema["format"]
                files = []
                for file_id in json_dict[field_name]:
                    file_content = await get_content_by_file_id(
                        file_id,
                        format=format,
                        file_manager=file_manager,
                    )
                    files.append(file_content)

                result[field_name] = files

            elif list_type == "object":
                sub_file_result = []
                for file_dict in json_dict[field_name]:
                    sub_file = await parse_json_request(sub_class, file_dict, file_manager)
                    if len(sub_file) > 0:
                        sub_file_result.append(sub_file)

                if len(sub_file_result) > 0:
                    result[field_name] = sub_file_result

        if field_name in result:
            json_dict.pop(field_name, None)
        elif field_name not in result and field_name in json_dict:
            result[field_name] = json_dict.pop(field_name)

        fixed_value = model_field.json_schema_extra.get("x-ebagent-fixed-value", None)
        if fixed_value:
            result[field_name] = fixed_value
    result.update(json_dict)
    return result


@no_type_check
async def parse_json_response(
    view: Type[ToolParameterView], json_dict, file_manager: FileManager, file_metadata: Dict[str, str]
) -> Dict[str, Any]:
    result = {}
    for field_name, model_field in view.model_fields.items():
        list_type = get_typing_list_type(model_field.annotation)

        if list_type is None and model_field.json_schema_extra is not None:
            if not isinstance(model_field.json_schema_extra, dict):
                raise RemoteToolError(
                    f"`json_schema_extra` field must be dict type, but got "
                    f"<{type(model_field.json_schema_extra)}> in model_field<{field_name}>. "
                    "Please check the format of yaml in current tool."
                )

            if model_field.annotation == str and "x-ebagent-file-mime-type" in model_field.json_schema_extra:
                format = model_field.json_schema_extra.get("format", None)
                mime_type = model_field.json_schema_extra.get("x-ebagent-file-mime-type", None)

                if format is not None and mime_type is not None:
                    file = await create_file_from_data(
                        json_dict[field_name],
                        format=format,
                        mime_type=mime_type,
                        file_manager=file_manager,
                        file_metadata=file_metadata,
                    )
                    result[field_name] = file.id
            elif issubclass(model_field.annotation, ToolParameterView):
                files = await parse_json_response(
                    model_field.annotation,
                    json_dict[field_name],
                    file_manager=file_manager,
                    file_metadata=file_metadata,
                )
                if len(files) > 0:
                    result[field_name] = files

        else:
            array_json_schema = model_field.json_schema_extra.get("array_items_schema", None)
            sub_class = get_args(model_field.annotation)[0]
            if (
                list_type == "string"
                and array_json_schema is not None
                and array_json_schema.get("x-ebagent-file-mime-type", None)
            ):
                format = array_json_schema["format"]
                mime_type = array_json_schema["x-ebagent-file-mime-type"]
                files = []
                for file_content in json_dict[field_name]:
                    file = await create_file_from_data(
                        file_content,
                        format=format,
                        mime_type=mime_type,
                        file_manager=file_manager,
                        file_metadata=file_metadata,
                    )
                    files.append(file.id)
                result[field_name] = files
            elif list_type == "object":
                sub_file_result = []
                for file_dict in json_dict[field_name]:
                    sub_file = await parse_json_response(sub_class, file_dict, file_manager, file_metadata)
                    if len(sub_file) > 0:
                        sub_file_result.append(sub_file)

                if len(sub_file_result) > 0:
                    result[field_name] = sub_file_result

        if field_name in result:
            json_dict.pop(field_name, None)
        elif field_name not in result and field_name in json_dict:
            result[field_name] = json_dict.pop(field_name)

        fixed_value = model_field.json_schema_extra.get("x-ebagent-fixed-value", None)
        if fixed_value:
            result[field_name] = fixed_value

    result.update(json_dict)
    return result


async def parse_response(
    response: Response,
    file_manager: FileManager,
    file_metadata: Dict[str, str] = {},
    tool_parameter_view: Optional[Type[ToolParameterView]] = None,
) -> Dict[str, Any]:
    # 1. parse file by `Content-Disposition`
    content_disposition = response.headers.get("Content-Disposition", None)
    if content_disposition is not None:
        file_name = response.headers["Content-Disposition"].split("filename=")[1]
        local_file = await file_manager.create_file_from_bytes(
            response.content, file_name, file_purpose="assistants_output", file_metadata=file_metadata
        )
        return {"file": local_file.id}

    if is_json_response(response):
        if tool_parameter_view is None:
            return response.json()

        return await parse_json_response(
            tool_parameter_view, response.json(), file_manager=file_manager, file_metadata=file_metadata
        )

    # 3. parse file by content_type
    content_type = response.headers.get("Content-Type", None)
    if content_type is not None:
        file_suffix = get_file_suffix(content_type)
        local_file = await file_manager.create_file_from_bytes(
            response.content,
            f"tool-{file_suffix}",
            file_purpose="assistants_output",
            file_metadata=file_metadata,
        )
        return {"file": local_file.id}

    raise RemoteToolError(
        "Can not parse file from response: the type of data from response is not json, "
        "and can not find `Content-Disposition` or `Content-Type` field from response header.",
        stage="Output parsing",
    )


def get_fastapi_openapi(app):
    """get openapi dict of fastapi application

    refer to: https://github.com/tiangolo/fastapi/issues/3424#issuecomment-1283484665

    Args:
        app (FastAPI): the instance of fastapi application
    """
    fastapi = import_module(
        "fastapi",
        "Could not import fastapi or uvicorn python package. Please install it "
        "with `pip install fastapi`.",
    )

    if not app.openapi_schema:
        app.openapi_schema = fastapi.openapi.utils.get_openapi(
            title=app.title,
            version=app.version,
            openapi_version=app.openapi_version,
            description=app.description,
            terms_of_service=app.terms_of_service,
            contact=app.contact,
            license_info=app.license_info,
            routes=app.routes,
            tags=app.openapi_tags,
            servers=app.servers,
        )

        # remove validation response
        for _, method_item in app.openapi_schema.get("paths", {}).items():
            for _, param in method_item.items():
                responses = param.get("responses")
                # remove 422 response, also can remove other status code
                if "422" in responses:
                    del responses["422"]

        # remove Validation Schema
        schemas = deepcopy(app.openapi_schema["components"]["schemas"])
        for key in list(schemas.keys()):
            if "ValidationError" in key:
                schemas.pop(key)

        app.openapi_schema["components"]["schemas"] = schemas
