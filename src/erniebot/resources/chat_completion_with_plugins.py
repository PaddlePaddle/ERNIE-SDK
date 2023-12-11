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
from erniebot.response import EBResponse
from erniebot.types import ConfigDictType, HeadersType, RequestWithStream
from erniebot.utils.misc import NOT_GIVEN, NotGiven, filter_args

from .abc import CreatableWithStreaming
from .resource import EBResource

__all__ = ["ChatCompletionWithPlugins"]


class ChatCompletionWithPlugins(EBResource, CreatableWithStreaming):
    SUPPORTED_API_TYPES: ClassVar[Tuple[APIType, ...]] = (
        APIType.QIANFAN,
        APIType.CUSTOM,
    )

    @overload
    @classmethod
    def create(
        cls,
        messages: List[dict],
        plugins: List[str],
        *,
        user_id: Union[str, NotGiven] = ...,
        stream: Union[Literal[False], NotGiven] = ...,
        headers: Optional[HeadersType] = ...,
        request_timeout: Optional[float] = ...,
        _config_: Optional[ConfigDictType] = ...,
    ) -> EBResponse:
        ...

    @overload
    @classmethod
    def create(
        cls,
        messages: List[dict],
        plugins: List[str],
        *,
        user_id: Union[str, NotGiven] = ...,
        stream: Literal[True],
        headers: Optional[HeadersType] = ...,
        request_timeout: Optional[float] = ...,
        _config_: Optional[ConfigDictType] = ...,
    ) -> Iterator[EBResponse]:
        ...

    @overload
    @classmethod
    def create(
        cls,
        messages: List[dict],
        plugins: List[str],
        *,
        user_id: Union[str, NotGiven] = ...,
        stream: bool,
        headers: Optional[HeadersType] = ...,
        request_timeout: Optional[float] = ...,
        _config_: Optional[ConfigDictType] = ...,
    ) -> Union[EBResponse, Iterator[EBResponse]]:
        ...

    @classmethod
    def create(
        cls,
        messages: List[dict],
        plugins: List[str],
        *,
        user_id: Union[str, NotGiven] = NOT_GIVEN,
        stream: Union[bool, NotGiven] = NOT_GIVEN,
        headers: Optional[HeadersType] = None,
        request_timeout: Optional[float] = None,
        _config_: Optional[ConfigDictType] = None,
    ) -> Union[EBResponse, Iterator[EBResponse]]:
        config = _config_ or {}
        resource = cls(**config)
        kwargs = filter_args(
            messages=messages,
            plugins=plugins,
            user_id=user_id,
            stream=stream,
        )
        if headers is not None:
            kwargs["headers"] = headers
        if request_timeout is not None:
            kwargs["request_timeout"] = request_timeout
        resp = resource.create_resource(**kwargs)
        return resp

    @overload
    @classmethod
    async def acreate(
        cls,
        messages: List[dict],
        plugins: List[str],
        *,
        user_id: Union[str, NotGiven] = ...,
        stream: Union[Literal[False], NotGiven] = ...,
        headers: Optional[HeadersType] = ...,
        request_timeout: Optional[float] = ...,
        _config_: Optional[ConfigDictType] = ...,
    ) -> EBResponse:
        ...

    @overload
    @classmethod
    async def acreate(
        cls,
        messages: List[dict],
        plugins: List[str],
        *,
        user_id: Union[str, NotGiven] = ...,
        stream: Literal[True],
        headers: Optional[HeadersType] = ...,
        request_timeout: Optional[float] = ...,
        _config_: Optional[ConfigDictType] = ...,
    ) -> AsyncIterator[EBResponse]:
        ...

    @overload
    @classmethod
    async def acreate(
        cls,
        messages: List[dict],
        plugins: List[str],
        *,
        user_id: Union[str, NotGiven] = ...,
        stream: bool,
        headers: Optional[HeadersType] = ...,
        request_timeout: Optional[float] = ...,
        _config_: Optional[ConfigDictType] = ...,
    ) -> Union[EBResponse, AsyncIterator[EBResponse]]:
        ...

    @classmethod
    async def acreate(
        cls,
        messages: List[dict],
        plugins: List[str],
        *,
        user_id: Union[str, NotGiven] = NOT_GIVEN,
        stream: Union[bool, NotGiven] = NOT_GIVEN,
        headers: Optional[HeadersType] = None,
        request_timeout: Optional[float] = None,
        _config_: Optional[ConfigDictType] = None,
    ) -> Union[EBResponse, AsyncIterator[EBResponse]]:
        config = _config_ or {}
        resource = cls(**config)
        kwargs = filter_args(
            messages=messages,
            plugins=plugins,
            user_id=user_id,
            stream=stream,
        )
        if headers is not None:
            kwargs["headers"] = headers
        if request_timeout is not None:
            kwargs["request_timeout"] = request_timeout
        resp = await resource.acreate_resource(**kwargs)
        return resp

    def _prepare_create(self, kwargs: Dict[str, Any]) -> RequestWithStream:
        def _set_val_if_key_exists(src: dict, dst: dict, key: str) -> None:
            if key in src:
                dst[key] = src[key]

        valid_keys = {
            "messages",
            "plugins",
            "user_id",
            "stream",
            "headers",
            "request_timeout",
        }

        invalid_keys = kwargs.keys() - valid_keys

        if len(invalid_keys) > 0:
            raise ValueError(f"Invalid keys found in `kwargs`: {list(invalid_keys)}")

        # messages
        if "messages" not in kwargs:
            raise errors.ArgumentNotFoundError("`messages` is not found.")
        messages = kwargs["messages"]

        # plugins
        if "plugins" not in kwargs:
            raise errors.ArgumentNotFoundError("`plugins` is not found.")
        plugins = kwargs["plugins"]

        # path
        if self.api_type in self.SUPPORTED_API_TYPES:
            path = "/erniebot/plugins"
        else:
            raise errors.UnsupportedAPITypeError(
                f"Supported API types: {self.get_supported_api_type_names()}"
            )

        # params
        params = {}
        params["messages"] = messages
        params["plugins"] = plugins
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
