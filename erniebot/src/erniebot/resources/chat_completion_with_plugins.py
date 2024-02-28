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

import erniebot.errors as errors
from erniebot.api_types import APIType
from erniebot.types import ConfigDictType, HeadersType, RequestWithStream
from erniebot.utils.misc import NOT_GIVEN, NotGiven, filter_args, transform

from .abc import CreatableWithStreaming
from .chat_completion import ChatCompletionResponse
from .resource import EBResource

__all__ = ["ChatCompletionWithPlugins"]


class ChatCompletionWithPlugins(EBResource, CreatableWithStreaming):
    SUPPORTED_API_TYPES: ClassVar[Tuple[APIType, ...]] = (
        APIType.QIANFAN,
        APIType.CUSTOM,
        APIType.AISTUDIO,
    )
    _API_INFO_DICT: ClassVar[Dict[APIType, Dict[str, Any]]] = {
        APIType.QIANFAN: {
            "path": "/erniebot/plugins",
        },
        APIType.CUSTOM: {
            "path": "/erniebot/plugins_v3",
        },
        APIType.AISTUDIO: {
            "path": "/erniebot/plugins",
        },
    }

    @overload
    @classmethod
    def create(
        cls,
        messages: List[dict],
        plugins: List[str],
        *,
        functions: Union[List[dict], NotGiven] = ...,
        user_id: Union[str, NotGiven] = ...,
        stream: Union[Literal[False], NotGiven] = ...,
        extra_params: Optional[dict] = ...,
        headers: Optional[HeadersType] = ...,
        request_timeout: Optional[float] = ...,
        _config_: Optional[ConfigDictType] = ...,
    ) -> ChatCompletionResponse:
        ...

    @overload
    @classmethod
    def create(
        cls,
        messages: List[dict],
        plugins: List[str],
        *,
        functions: Union[List[dict], NotGiven] = ...,
        user_id: Union[str, NotGiven] = ...,
        stream: Literal[True],
        extra_params: Optional[dict] = ...,
        headers: Optional[HeadersType] = ...,
        request_timeout: Optional[float] = ...,
        _config_: Optional[ConfigDictType] = ...,
    ) -> Iterator[ChatCompletionResponse]:
        ...

    @overload
    @classmethod
    def create(
        cls,
        messages: List[dict],
        plugins: List[str],
        *,
        functions: Union[List[dict], NotGiven] = ...,
        user_id: Union[str, NotGiven] = ...,
        stream: bool,
        headers: Optional[HeadersType] = ...,
        request_timeout: Optional[float] = ...,
        _config_: Optional[ConfigDictType] = ...,
    ) -> Union[ChatCompletionResponse, Iterator[ChatCompletionResponse]]:
        ...

    @classmethod
    def create(
        cls,
        messages: List[dict],
        plugins: List[str],
        *,
        functions: Union[List[dict], NotGiven] = NOT_GIVEN,
        user_id: Union[str, NotGiven] = NOT_GIVEN,
        stream: Union[bool, NotGiven] = NOT_GIVEN,
        extra_params: Optional[dict] = None,
        headers: Optional[HeadersType] = None,
        request_timeout: Optional[float] = None,
        _config_: Optional[ConfigDictType] = None,
    ) -> Union[ChatCompletionResponse, Iterator[ChatCompletionResponse]]:
        config = _config_ or {}
        resource = cls(**config)
        kwargs = filter_args(
            messages=messages,
            functions=functions,
            plugins=plugins,
            user_id=user_id,
            stream=stream,
        )
        if extra_params is not None:
            kwargs["extra_params"] = extra_params
        if headers is not None:
            kwargs["headers"] = headers
        if request_timeout is not None:
            kwargs["request_timeout"] = request_timeout
        resp = resource.create_resource(**kwargs)
        return transform(ChatCompletionResponse.from_mapping, resp)

    @overload
    @classmethod
    async def acreate(
        cls,
        messages: List[dict],
        plugins: List[str],
        *,
        functions: Union[List[dict], NotGiven] = ...,
        user_id: Union[str, NotGiven] = ...,
        stream: Union[Literal[False], NotGiven] = ...,
        extra_params: Optional[dict] = ...,
        headers: Optional[HeadersType] = ...,
        request_timeout: Optional[float] = ...,
        _config_: Optional[ConfigDictType] = ...,
    ) -> ChatCompletionResponse:
        ...

    @overload
    @classmethod
    async def acreate(
        cls,
        messages: List[dict],
        plugins: List[str],
        *,
        functions: Union[List[dict], NotGiven] = ...,
        user_id: Union[str, NotGiven] = ...,
        stream: Literal[True],
        headers: Optional[HeadersType] = ...,
        request_timeout: Optional[float] = ...,
        _config_: Optional[ConfigDictType] = ...,
    ) -> AsyncIterator[ChatCompletionResponse]:
        ...

    @overload
    @classmethod
    async def acreate(
        cls,
        messages: List[dict],
        plugins: List[str],
        *,
        functions: Union[List[dict], NotGiven] = ...,
        user_id: Union[str, NotGiven] = ...,
        stream: bool,
        extra_params: Optional[dict] = ...,
        headers: Optional[HeadersType] = ...,
        request_timeout: Optional[float] = ...,
        _config_: Optional[ConfigDictType] = ...,
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionResponse]]:
        ...

    @classmethod
    async def acreate(
        cls,
        messages: List[dict],
        plugins: List[str],
        *,
        functions: Union[List[dict], NotGiven] = NOT_GIVEN,
        user_id: Union[str, NotGiven] = NOT_GIVEN,
        stream: Union[bool, NotGiven] = NOT_GIVEN,
        extra_params: Optional[dict] = None,
        headers: Optional[HeadersType] = None,
        request_timeout: Optional[float] = None,
        _config_: Optional[ConfigDictType] = None,
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionResponse]]:
        config = _config_ or {}
        resource = cls(**config)
        kwargs = filter_args(
            messages=messages,
            plugins=plugins,
            functions=functions,
            user_id=user_id,
            stream=stream,
        )
        if extra_params is not None:
            kwargs["extra_params"] = extra_params
        if headers is not None:
            kwargs["headers"] = headers
        if request_timeout is not None:
            kwargs["request_timeout"] = request_timeout
        resp = await resource.acreate_resource(**kwargs)
        return transform(ChatCompletionResponse.from_mapping, resp)

    def _prepare_create(self, kwargs: Dict[str, Any]) -> RequestWithStream:
        def _set_val_if_key_exists(src: dict, dst: dict, key: str) -> None:
            if key in src:
                dst[key] = src[key]

        valid_keys = {
            "messages",
            "plugins",
            "functions",
            "user_id",
            "stream",
            "extra_params",
            "headers",
            "request_timeout",
        }

        invalid_keys = kwargs.keys() - valid_keys

        if len(invalid_keys) > 0:
            raise ValueError(f"Invalid keys found in `kwargs`: {list(invalid_keys)}")

        # messages
        if "messages" not in kwargs:
            raise errors.ArgumentNotFoundError("messages")
        messages = kwargs["messages"]

        # plugins
        if "plugins" not in kwargs:
            raise errors.ArgumentNotFoundError("plugins")
        plugins = kwargs["plugins"]

        # path
        if self.api_type in self.SUPPORTED_API_TYPES:
            api_info = self._API_INFO_DICT[self.api_type]
            path = api_info["path"]
        else:
            raise errors.UnsupportedAPITypeError(
                f"Supported API types: {self.get_supported_api_type_names()}"
            )

        # params
        params = {}
        params["messages"] = messages
        params["plugins"] = plugins
        _set_val_if_key_exists(kwargs, params, "functions")
        if self.api_type is not APIType.AISTUDIO:
            _set_val_if_key_exists(kwargs, params, "user_id")
        _set_val_if_key_exists(kwargs, params, "stream")
        if "extra_params" in kwargs:
            params.update(kwargs["extra_params"])

        # headers
        headers: HeadersType = {}
        if self.api_type is APIType.AISTUDIO:
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
