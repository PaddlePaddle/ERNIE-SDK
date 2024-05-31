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

from typing import (
    Any,
    AsyncIterator,
    ClassVar,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    overload,
)

import jsonschema
import jsonschema.exceptions

import erniebot.errors as errors
from erniebot.api_types import APIType
from erniebot.response import EBResponse
from erniebot.types import ConfigDictType, HeadersType, RequestWithStream
from erniebot.utils import logging
from erniebot.utils.misc import NOT_GIVEN, NotGiven, filter_args, transform

from .abc import CreatableWithStreaming
from .resource import EBResource

__all__ = ["ChatCompletion", "ChatCompletionResponse"]


class ChatCompletion(EBResource, CreatableWithStreaming):
    SUPPORTED_API_TYPES: ClassVar[Tuple[APIType, ...]] = (
        APIType.QIANFAN,
        APIType.AISTUDIO,
        APIType.CUSTOM,
    )
    _API_INFO_DICT: ClassVar[Dict[APIType, Dict[str, Any]]] = {
        APIType.QIANFAN: {
            "resource_id": "chat",
            "models": {
                "ernie-3.5": {
                    "model_id": "completions",
                },
                "ernie-3.5-8k": {
                    "model_id": "completions",
                },
                "ernie-3.5-8k-0205": {
                    "model_id": "ernie-3.5-8k-0205",
                },
                "ernie-3.5-8k-0329": {
                    "model_id": "ernie-3.5-8k-0329",
                },
                "ernie-3.5-128k": {
                    "model_id": "ernie-3.5-128k",
                },
                "ernie-lite": {
                    "model_id": "eb-instant",
                },
                "ernie-lite-8k-0308": {
                    "model_id": "ernie-lite-8k",
                },
                "ernie-4.0": {
                    "model_id": "completions_pro",
                },
                "ernie-4.0-8k-0329": {
                    "model_id": "ernie-4.0-8k-0329",
                },
                "ernie-4.0-8k-0104": {
                    "model_id": "ernie-4.0-8k-0104",
                },
                "ernie-speed": {
                    "model_id": "ernie_speed",
                },
                "ernie-speed-128k": {
                    "model_id": "ernie-speed-128k",
                },
                "ernie-tiny-8k": {
                    "model_id": "ernie-tiny-8k",
                },
                "ernie-char-8k": {
                    "model_id": "ernie-char-8k",
                },
            },
        },
        APIType.AISTUDIO: {
            "resource_id": "chat",
            "models": {
                "ernie-3.5": {
                    "model_id": "completions",
                },
                "ernie-3.5-8k": {
                    "model_id": "completions",
                },
                "ernie-lite": {
                    "model_id": "eb-instant",
                },
                "ernie-4.0": {
                    "model_id": "completions_pro",
                },
                "ernie-speed": {
                    "model_id": "ernie_speed",
                },
                "ernie-speed-128k": {
                    "model_id": "ernie-speed-128k",
                },
                "ernie-tiny-8k": {
                    "model_id": "ernie-tiny-8k",
                },
                "ernie-char-8k": {
                    "model_id": "ernie-char-8k",
                },
            },
        },
        APIType.CUSTOM: {
            "resource_id": "chat",
            "models": {
                "ernie-3.5": {
                    "model_id": "completions",
                },
                "ernie-4.0": {
                    "model_id": "completions_pro",
                },
                "ernie-longtext": {
                    "model_id": "completions",
                },
                "ernie-speed": {
                    "model_id": "ernie_speed",
                },
            },
        },
    }

    @overload
    @classmethod
    def create(
        cls,
        model: str,
        messages: List[dict],
        *,
        functions: Union[List[dict], NotGiven] = ...,
        temperature: Union[float, NotGiven] = ...,
        top_p: Union[float, NotGiven] = ...,
        penalty_score: Union[float, NotGiven] = ...,
        system: Union[str, NotGiven] = ...,
        stop: Union[str, NotGiven] = ...,
        disable_search: Union[bool, NotGiven] = ...,
        enable_citation: Union[bool, NotGiven] = ...,
        user_id: Union[str, NotGiven] = ...,
        tool_choice: Union[dict, NotGiven] = ...,
        stream: Union[Literal[False], NotGiven] = ...,
        validate_functions: bool = ...,
        extra_params: Optional[dict] = ...,
        headers: Optional[HeadersType] = ...,
        request_timeout: Optional[float] = ...,
        response_format: Optional[Literal["json_object", "text"]] = ...,
        max_output_tokens: Optional[int] = ...,
        _config_: Optional[ConfigDictType] = ...,
    ) -> "ChatCompletionResponse":
        ...

    @overload
    @classmethod
    def create(
        cls,
        model: str,
        messages: List[dict],
        *,
        functions: Union[List[dict], NotGiven] = ...,
        temperature: Union[float, NotGiven] = ...,
        top_p: Union[float, NotGiven] = ...,
        penalty_score: Union[float, NotGiven] = ...,
        system: Union[str, NotGiven] = ...,
        stop: Union[str, NotGiven] = ...,
        disable_search: Union[bool, NotGiven] = ...,
        enable_citation: Union[bool, NotGiven] = ...,
        user_id: Union[str, NotGiven] = ...,
        tool_choice: Union[dict, NotGiven] = ...,
        stream: Literal[True],
        validate_functions: bool = ...,
        extra_params: Optional[dict] = ...,
        headers: Optional[HeadersType] = ...,
        request_timeout: Optional[float] = ...,
        response_format: Optional[Literal["json_object", "text"]] = ...,
        max_output_tokens: Optional[int] = ...,
        _config_: Optional[ConfigDictType] = ...,
    ) -> Iterator["ChatCompletionResponse"]:
        ...

    @overload
    @classmethod
    def create(
        cls,
        model: str,
        messages: List[dict],
        *,
        functions: Union[List[dict], NotGiven] = ...,
        temperature: Union[float, NotGiven] = ...,
        top_p: Union[float, NotGiven] = ...,
        penalty_score: Union[float, NotGiven] = ...,
        system: Union[str, NotGiven] = ...,
        stop: Union[str, NotGiven] = ...,
        disable_search: Union[bool, NotGiven] = ...,
        enable_citation: Union[bool, NotGiven] = ...,
        user_id: Union[str, NotGiven] = ...,
        tool_choice: Union[dict, NotGiven] = ...,
        stream: bool,
        validate_functions: bool = ...,
        extra_params: Optional[dict] = ...,
        headers: Optional[HeadersType] = ...,
        request_timeout: Optional[float] = ...,
        response_format: Optional[Literal["json_object", "text"]] = ...,
        max_output_tokens: Optional[int] = ...,
        _config_: Optional[ConfigDictType] = ...,
    ) -> Union["ChatCompletionResponse", Iterator["ChatCompletionResponse"]]:
        ...

    @classmethod
    def create(
        cls,
        model: str,
        messages: List[dict],
        *,
        functions: Union[List[dict], NotGiven] = NOT_GIVEN,
        temperature: Union[float, NotGiven] = NOT_GIVEN,
        top_p: Union[float, NotGiven] = NOT_GIVEN,
        penalty_score: Union[float, NotGiven] = NOT_GIVEN,
        system: Union[str, NotGiven] = NOT_GIVEN,
        stop: Union[str, NotGiven] = NOT_GIVEN,
        disable_search: Union[bool, NotGiven] = NOT_GIVEN,
        enable_citation: Union[bool, NotGiven] = NOT_GIVEN,
        user_id: Union[str, NotGiven] = NOT_GIVEN,
        tool_choice: Union[dict, NotGiven] = NOT_GIVEN,
        stream: Union[bool, NotGiven] = NOT_GIVEN,
        validate_functions: bool = False,
        extra_params: Optional[dict] = None,
        headers: Optional[HeadersType] = None,
        request_timeout: Optional[float] = None,
        response_format: Optional[Literal["json_object", "text"]] = None,
        max_output_tokens: Optional[int] = None,
        _config_: Optional[ConfigDictType] = None,
    ) -> Union["ChatCompletionResponse", Iterator["ChatCompletionResponse"]]:
        """Creates a model response for the given conversation.

        Args:
            model: Name of the model to use.
            messages: Messages comprising the conversation so far.
            functions: Descriptions of the functions that the model may generate
                JSON inputs for.
            temperature: Sampling temperature to use.
            top_p: Parameter of nucleus sampling that affects the diversity of
                generated content.
            penalty_score: Penalty assigned to new tokens that appear in the
                generated text so far.
            system: Text that tells the model how to interpret the conversation.
            stop: Instructs the model to stop generating further tokens at the
                first occurrence of any of these strings.
            disable_search: Whether to disable the search engine.
            enable_citation: Whether to enable citation generation.
            user_id: ID for the end user.
            stream: Whether to enable response streaming.
            validate_functions: Whether to validate the function descriptions.
            headers: Custom headers to send with the request.
            request_timeout: Timeout for a single request.
            _config_: Overrides the global settings.

        Returns:
            If `stream` is True, returns an iterator that yields response
            objects. Otherwise returns a response object.
        """
        config = _config_ or {}
        resource = cls(**config)
        kwargs = filter_args(
            model=model,
            messages=messages,
            functions=functions,
            temperature=temperature,
            top_p=top_p,
            penalty_score=penalty_score,
            system=system,
            stop=stop,
            disable_search=disable_search,
            enable_citation=enable_citation,
            user_id=user_id,
            tool_choice=tool_choice,
            stream=stream,
            max_output_tokens=max_output_tokens,
        )
        kwargs["validate_functions"] = validate_functions
        if extra_params is not None:
            kwargs["extra_params"] = extra_params
        if headers is not None:
            kwargs["headers"] = headers
        if request_timeout is not None:
            kwargs["request_timeout"] = request_timeout
        if response_format is not None:
            kwargs["response_format"] = response_format

        resp = resource.create_resource(**kwargs)
        return transform(ChatCompletionResponse.from_mapping, resp)

    @overload
    @classmethod
    async def acreate(
        cls,
        model: str,
        messages: List[dict],
        *,
        functions: Union[List[dict], NotGiven] = ...,
        temperature: Union[float, NotGiven] = ...,
        top_p: Union[float, NotGiven] = ...,
        penalty_score: Union[float, NotGiven] = ...,
        system: Union[str, NotGiven] = ...,
        stop: Union[str, NotGiven] = ...,
        disable_search: Union[bool, NotGiven] = ...,
        enable_citation: Union[bool, NotGiven] = ...,
        user_id: Union[str, NotGiven] = ...,
        tool_choice: Union[dict, NotGiven] = ...,
        stream: Union[Literal[False], NotGiven] = ...,
        validate_functions: bool = ...,
        extra_params: Optional[dict] = ...,
        headers: Optional[HeadersType] = ...,
        request_timeout: Optional[float] = ...,
        response_format: Optional[Literal["json_object", "text"]] = ...,
        max_output_tokens: Optional[int] = ...,
        _config_: Optional[ConfigDictType] = ...,
    ) -> EBResponse:
        ...

    @overload
    @classmethod
    async def acreate(
        cls,
        model: str,
        messages: List[dict],
        *,
        functions: Union[List[dict], NotGiven] = ...,
        temperature: Union[float, NotGiven] = ...,
        top_p: Union[float, NotGiven] = ...,
        penalty_score: Union[float, NotGiven] = ...,
        system: Union[str, NotGiven] = ...,
        stop: Union[str, NotGiven] = ...,
        disable_search: Union[bool, NotGiven] = ...,
        enable_citation: Union[bool, NotGiven] = ...,
        user_id: Union[str, NotGiven] = ...,
        tool_choice: Union[dict, NotGiven] = ...,
        stream: Literal[True],
        validate_functions: bool = ...,
        extra_params: Optional[dict] = ...,
        headers: Optional[HeadersType] = ...,
        request_timeout: Optional[float] = ...,
        response_format: Optional[Literal["json_object", "text"]] = ...,
        max_output_tokens: Optional[int] = ...,
        _config_: Optional[ConfigDictType] = ...,
    ) -> AsyncIterator["ChatCompletionResponse"]:
        ...

    @overload
    @classmethod
    async def acreate(
        cls,
        model: str,
        messages: List[dict],
        *,
        functions: Union[List[dict], NotGiven] = ...,
        temperature: Union[float, NotGiven] = ...,
        top_p: Union[float, NotGiven] = ...,
        penalty_score: Union[float, NotGiven] = ...,
        system: Union[str, NotGiven] = ...,
        stop: Union[str, NotGiven] = ...,
        disable_search: Union[bool, NotGiven] = ...,
        enable_citation: Union[bool, NotGiven] = ...,
        user_id: Union[str, NotGiven] = ...,
        tool_choice: Union[dict, NotGiven] = ...,
        stream: bool,
        validate_functions: bool = ...,
        extra_params: Optional[dict] = ...,
        headers: Optional[HeadersType] = ...,
        request_timeout: Optional[float] = ...,
        response_format: Optional[Literal["json_object", "text"]] = ...,
        max_output_tokens: Optional[int] = ...,
        _config_: Optional[ConfigDictType] = ...,
    ) -> Union["ChatCompletionResponse", AsyncIterator["ChatCompletionResponse"]]:
        ...

    @classmethod
    async def acreate(
        cls,
        model: str,
        messages: List[dict],
        *,
        functions: Union[List[dict], NotGiven] = NOT_GIVEN,
        temperature: Union[float, NotGiven] = NOT_GIVEN,
        top_p: Union[float, NotGiven] = NOT_GIVEN,
        penalty_score: Union[float, NotGiven] = NOT_GIVEN,
        system: Union[str, NotGiven] = NOT_GIVEN,
        stop: Union[str, NotGiven] = NOT_GIVEN,
        disable_search: Union[bool, NotGiven] = NOT_GIVEN,
        enable_citation: Union[bool, NotGiven] = NOT_GIVEN,
        user_id: Union[str, NotGiven] = NOT_GIVEN,
        tool_choice: Union[dict, NotGiven] = NOT_GIVEN,
        stream: Union[bool, NotGiven] = NOT_GIVEN,
        validate_functions: bool = False,
        extra_params: Optional[dict] = None,
        headers: Optional[HeadersType] = None,
        request_timeout: Optional[float] = None,
        response_format: Optional[Literal["json_object", "text"]] = None,
        max_output_tokens: Optional[int] = None,
        _config_: Optional[ConfigDictType] = None,
    ) -> Union["ChatCompletionResponse", AsyncIterator["ChatCompletionResponse"]]:
        """Creates a model response for the given conversation.

        Args:
            model: Name of the model to use.
            messages: Messages comprising the conversation so far.
            functions: Descriptions of the functions that the model may generate
                JSON inputs for.
            temperature: Sampling temperature to use.
            top_p: Parameter of nucleus sampling that affects the diversity of
                generated content.
            penalty_score: Penalty assigned to new tokens that appear in the
                generated text so far.
            system: Text that tells the model how to interpret the conversation.
            stop: Instructs the model to stop generating further tokens at the
                first occurrence of any of these strings.
            disable_search: Whether to disable the search engine.
            enable_citation: Whether to enable citation generation.
            user_id: ID for the end user.
            stream: Whether to enable response streaming.
            validate_functions: Whether to validate the function descriptions.
            headers: Custom headers to send with the request.
            request_timeout: Timeout for a single request.
            response_format: Format of the response.
            _config_: Overrides the global settings.

        Returns:
            If `stream` is True, returns an iterator that yields response
            objects. Otherwise returns a response object.
        """
        config = _config_ or {}
        resource = cls(**config)
        kwargs = filter_args(
            model=model,
            messages=messages,
            functions=functions,
            temperature=temperature,
            top_p=top_p,
            penalty_score=penalty_score,
            system=system,
            stop=stop,
            disable_search=disable_search,
            enable_citation=enable_citation,
            user_id=user_id,
            tool_choice=tool_choice,
            stream=stream,
            max_output_tokens=max_output_tokens,
        )
        kwargs["validate_functions"] = validate_functions
        if extra_params is not None:
            kwargs["extra_params"] = extra_params
        if headers is not None:
            kwargs["headers"] = headers
        if request_timeout is not None:
            kwargs["request_timeout"] = request_timeout

        resp = await resource.acreate_resource(**kwargs)
        return transform(ChatCompletionResponse.from_mapping, resp)

    def _check_model_kwargs(self, model_name: str, kwargs: Dict[str, Any]) -> None:
        if model_name in ("ernie-speed", "ernie-speed-128k", "ernie-char-8k", "ernie-tiny-8k", "ernie-lite"):
            for arg in ("functions", "disable_search", "enable_citation", "tool_choice", "response_format"):
                if arg in kwargs:
                    raise errors.InvalidArgumentError(
                        f"`{arg}` is not supported by the `{model_name}` model."
                    )

    def _prepare_create(self, kwargs: Dict[str, Any]) -> RequestWithStream:
        def _update_model_name(given_name: str, old_name_to_new_name: Dict[str, str]) -> str:
            if given_name in old_name_to_new_name:
                new_name = old_name_to_new_name[given_name]
                logging.warning(
                    "'%s' will be deprecated in the future. Please use '%s' instead.", given_name, new_name
                )
                return new_name
            else:
                return given_name

        def _set_val_if_key_exists(src: dict, dst: dict, key: str) -> None:
            if key in src:
                dst[key] = src[key]

        valid_keys = {
            "model",
            "messages",
            "functions",
            "temperature",
            "top_p",
            "penalty_score",
            "system",
            "stop",
            "disable_search",
            "enable_citation",
            "user_id",
            "tool_choice",
            "stream",
            "validate_functions",
            "extra_params",
            "headers",
            "request_timeout",
            "response_format",
            "max_output_tokens",
        }

        invalid_keys = kwargs.keys() - valid_keys

        if len(invalid_keys) > 0:
            raise ValueError(f"Invalid keys found in `kwargs`: {list(invalid_keys)}")

        # model
        if "model" not in kwargs:
            raise errors.ArgumentNotFoundError("model")
        model = kwargs["model"]
        # For backward compatibility
        model = _update_model_name(
            model,
            {
                "ernie-bot": "ernie-3.5",
                "ernie-bot-turbo": "ernie-lite",
                "ernie-bot-4": "ernie-4.0",
                "ernie-bot-8k": "ernie-3.5-8k",
                "ernie-longtext": "ernie-3.5-8k",
                "ernie-turbo": "ernie-lite",
            },
        )

        # messages
        if "messages" not in kwargs:
            raise errors.ArgumentNotFoundError("messages")
        messages = kwargs["messages"]

        # path
        if self.api_type in self.SUPPORTED_API_TYPES:
            api_info = self._API_INFO_DICT[self.api_type]
            if model not in api_info["models"]:
                raise errors.InvalidArgumentError(f"{repr(model)} is not a supported model.")
            path = f"/{api_info['resource_id']}/{api_info['models'][model]['model_id']}"
        else:
            raise errors.UnsupportedAPITypeError(
                f"Supported API types: {self.get_supported_api_type_names()}"
            )

        # params
        params = {}
        self._check_model_kwargs(model, kwargs)

        params["messages"] = messages
        if "functions" in kwargs:
            functions = kwargs["functions"]
            if kwargs.get("validate_functions", False):
                self._validate_functions(functions)
            params["functions"] = functions
        _set_val_if_key_exists(kwargs, params, "temperature")
        _set_val_if_key_exists(kwargs, params, "top_p")
        _set_val_if_key_exists(kwargs, params, "penalty_score")
        _set_val_if_key_exists(kwargs, params, "system")
        _set_val_if_key_exists(kwargs, params, "stop")
        _set_val_if_key_exists(kwargs, params, "disable_search")
        _set_val_if_key_exists(kwargs, params, "enable_citation")
        if self.api_type is not APIType.AISTUDIO:
            # The AI Studio backend automatically injects `user_id`.
            _set_val_if_key_exists(kwargs, params, "user_id")
        _set_val_if_key_exists(kwargs, params, "tool_choice")
        _set_val_if_key_exists(kwargs, params, "stream")
        _set_val_if_key_exists(kwargs, params, "max_output_tokens")
        _set_val_if_key_exists(kwargs, params, "response_format")

        if "extra_params" in kwargs:
            params.update(kwargs["extra_params"])

        # headers
        headers: HeadersType = {}
        if self.api_type is APIType.AISTUDIO or self.api_type is APIType.CUSTOM:
            headers["Content-Type"] = "application/json"
        if "headers" in kwargs:
            headers.update(kwargs["headers"])

        # request_timeout
        request_timeout = kwargs.get("request_timeout", None)

        # stream
        stream = kwargs.get("stream", False)

        return RequestWithStream(
            method="POST",
            path=path,
            params=params,
            headers=headers,
            timeout=request_timeout,
            stream=stream,
        )

    @classmethod
    def _validate_functions(cls, functions: List[dict]) -> None:
        for idx, function in enumerate(functions):
            if "parameters" in function:
                parameters = function["parameters"]
                if not cls._check_json_schema(parameters):
                    raise errors.InvalidArgumentError(
                        f"`parameters` of function {idx} is not a valid schema."
                    )
                if parameters == {} or parameters == {"type": "object"}:
                    raise errors.InvalidArgumentError(
                        "For empty parameters, please set `type` to 'object' and `properties` to {}."
                    )
            if "responses" in function:
                responses = function["responses"]
                if not cls._check_json_schema(responses):
                    raise errors.InvalidArgumentError(
                        f"`responses` of function {idx} is not a valid schema."
                    )

    @staticmethod
    def _check_json_schema(schema: dict) -> bool:
        try:
            jsonschema.Draft202012Validator.check_schema(schema)
        except jsonschema.exceptions.SchemaError:
            return False
        else:
            return True


class ChatCompletionResponse(EBResponse):
    @property
    def is_function_response(self) -> bool:
        return hasattr(self, "function_call")

    def get_result(self) -> Any:
        if self.is_function_response:
            return self.function_call
        else:
            return self.result

    def to_message(self) -> Dict[str, Any]:
        message: Dict[str, Any] = {"role": "assistant"}
        if self.is_function_response:
            message["content"] = None
            message["function_call"] = self.function_call
        else:
            message["content"] = self.result
        return message
