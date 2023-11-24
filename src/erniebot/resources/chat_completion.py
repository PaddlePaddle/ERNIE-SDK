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
from erniebot.utils.misc import filter_args, transform

from .abc import CreatableWithStreaming
from .resource import EBResource

__all__ = ["ChatCompletion", "ChatCompletionResponse"]


class ChatCompletion(EBResource, CreatableWithStreaming):
    SUPPORTED_API_TYPES: ClassVar[Tuple[APIType, ...]] = (
        APIType.QIANFAN,
        APIType.AISTUDIO,
    )
    _API_INFO_DICT: ClassVar[Dict[APIType, Dict[str, Any]]] = {
        APIType.QIANFAN: {
            "resource_id": "chat",
            "models": {
                "ernie-bot": {
                    "model_id": "completions",
                },
                "ernie-bot-turbo": {
                    "model_id": "eb-instant",
                },
                "ernie-bot-4": {
                    "model_id": "completions_pro",
                },
                "ernie-bot-8k": {
                    "model_id": "ernie_bot_8k",
                },
            },
        },
        APIType.AISTUDIO: {
            "resource_id": "chat",
            "models": {
                "ernie-bot": {
                    "model_id": "completions",
                },
                "ernie-bot-turbo": {
                    "model_id": "eb-instant",
                },
                "ernie-bot-4": {
                    "model_id": "completions_pro",
                },
                "ernie-bot-8k": {
                    "model_id": "ernie_bot_8k",
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
        functions: Optional[List[dict]] = ...,
        temperature: Optional[float] = ...,
        top_p: Optional[float] = ...,
        penalty_score: Optional[float] = ...,
        system: Optional[str] = ...,
        user_id: Optional[str] = ...,
        stream: Optional[Literal[False]] = ...,
        validate_functions: bool = ...,
        headers: Optional[HeadersType] = ...,
        request_timeout: Optional[float] = ...,
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
        functions: Optional[List[dict]] = ...,
        temperature: Optional[float] = ...,
        top_p: Optional[float] = ...,
        penalty_score: Optional[float] = ...,
        system: Optional[str] = ...,
        user_id: Optional[str] = ...,
        stream: Literal[True],
        validate_functions: bool = ...,
        headers: Optional[HeadersType] = ...,
        request_timeout: Optional[float] = ...,
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
        functions: Optional[List[dict]] = ...,
        temperature: Optional[float] = ...,
        top_p: Optional[float] = ...,
        penalty_score: Optional[float] = ...,
        system: Optional[str] = ...,
        user_id: Optional[str] = ...,
        stream: bool = ...,
        validate_functions: bool = ...,
        headers: Optional[HeadersType] = ...,
        request_timeout: Optional[float] = ...,
        _config_: Optional[ConfigDictType] = ...,
    ) -> Union["ChatCompletionResponse", Iterator["ChatCompletionResponse"]]:
        ...

    @classmethod
    def create(
        cls,
        model: str,
        messages: List[dict],
        *,
        functions: Optional[List[dict]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        penalty_score: Optional[float] = None,
        system: Optional[str] = None,
        user_id: Optional[str] = None,
        stream: Optional[bool] = None,
        validate_functions: bool = False,
        headers: Optional[HeadersType] = None,
        request_timeout: Optional[float] = None,
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
            user_id: ID for the end user.
            stream: Whether to enable response streaming.
            validate_functions: Whether to validate the function descriptions.
            headers: Additional headers to send with the request.
            request_timeout: Timeout for a single request.
            _config_: Overrides the global settings.

        Returns:
            If `stream` is True, returns an iterator that yields response
            objects. Otherwise returns a response object.
        """
        config = _config_ or {}
        resource = cls(**config)
        resp = resource.create_resource(
            **filter_args(
                model=model,
                messages=messages,
                functions=functions,
                temperature=temperature,
                top_p=top_p,
                penalty_score=penalty_score,
                system=system,
                user_id=user_id,
                stream=stream,
                validate_functions=validate_functions,
                headers=headers,
                request_timeout=request_timeout,
            )
        )
        return transform(ChatCompletionResponse.from_mapping, resp)

    @overload
    @classmethod
    async def acreate(
        cls,
        model: str,
        messages: List[dict],
        *,
        functions: Optional[List[dict]] = ...,
        temperature: Optional[float] = ...,
        top_p: Optional[float] = ...,
        penalty_score: Optional[float] = ...,
        system: Optional[str] = ...,
        user_id: Optional[str] = ...,
        stream: Optional[Literal[False]] = ...,
        validate_functions: bool = ...,
        headers: Optional[HeadersType] = ...,
        request_timeout: Optional[float] = ...,
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
        functions: Optional[List[dict]] = ...,
        temperature: Optional[float] = ...,
        top_p: Optional[float] = ...,
        penalty_score: Optional[float] = ...,
        system: Optional[str] = ...,
        user_id: Optional[str] = ...,
        stream: Literal[True],
        validate_functions: bool = ...,
        headers: Optional[HeadersType] = ...,
        request_timeout: Optional[float] = ...,
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
        functions: Optional[List[dict]] = ...,
        temperature: Optional[float] = ...,
        top_p: Optional[float] = ...,
        penalty_score: Optional[float] = ...,
        system: Optional[str] = ...,
        user_id: Optional[str] = ...,
        stream: bool = ...,
        validate_functions: bool = ...,
        headers: Optional[HeadersType] = ...,
        request_timeout: Optional[float] = ...,
        _config_: Optional[ConfigDictType] = ...,
    ) -> Union["ChatCompletionResponse", AsyncIterator["ChatCompletionResponse"]]:
        ...

    @classmethod
    async def acreate(
        cls,
        model: str,
        messages: List[dict],
        *,
        functions: Optional[List[dict]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        penalty_score: Optional[float] = None,
        system: Optional[str] = None,
        user_id: Optional[str] = None,
        stream: Optional[bool] = None,
        validate_functions: bool = False,
        headers: Optional[HeadersType] = None,
        request_timeout: Optional[float] = None,
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
            user_id: ID for the end user.
            stream: Whether to enable response streaming or not.
            validate_functions: Whether to validate the function descriptions.
            headers: Additional headers to send with the request.
            request_timeout: Timeout for a single request.
            _config_: Overrides the global settings.

        Returns:
            If `stream` is True, returns an iterator that yields response
            objects. Otherwise returns a response object.
        """
        config = _config_ or {}
        resource = cls(**config)
        resp = await resource.acreate_resource(
            **filter_args(
                model=model,
                messages=messages,
                functions=functions,
                temperature=temperature,
                top_p=top_p,
                penalty_score=penalty_score,
                system=system,
                user_id=user_id,
                stream=stream,
                validate_functions=validate_functions,
                headers=headers,
                request_timeout=request_timeout,
            )
        )
        return transform(ChatCompletionResponse.from_mapping, resp)

    def _prepare_create(self, kwargs: Dict[str, Any]) -> RequestWithStream:
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
            "user_id",
            "stream",
            "validate_functions",
            "headers",
            "request_timeout",
        }

        invalid_keys = kwargs.keys() - valid_keys

        if len(invalid_keys) > 0:
            raise ValueError(f"Invalid keys found in `kwargs`: {list(invalid_keys)}")

        # model
        if "model" not in kwargs:
            raise errors.ArgumentNotFoundError("`model` is not found.")
        model = kwargs["model"]
        # For backward compatibility
        if model == "ernie-bot-3.5":
            model = "ernie-bot"

        # messages
        if "messages" not in kwargs:
            raise errors.ArgumentNotFoundError("`messages` is not found.")
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
        if self.api_type is not APIType.AISTUDIO:
            # The AI Studio backend automatically injects `user_id`.
            _set_val_if_key_exists(kwargs, params, "user_id")
        _set_val_if_key_exists(kwargs, params, "stream")

        # headers
        headers = kwargs.get("headers", None)

        # request_timeout
        request_timeout = kwargs.get("request_timeout", None)

        # stream
        stream = kwargs.get("stream", False)

        return RequestWithStream(
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
