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

from typing import Any, ClassVar, Dict, List, Optional, Tuple

import erniebot.errors as errors
from erniebot.api_types import APIType
from erniebot.response import EBResponse
from erniebot.types import ConfigDictType, HeadersType, Request
from erniebot.utils.misc import filter_args

from .abc import Creatable
from .resource import EBResource

__all__ = ["ChatFile"]


class ChatFile(EBResource, Creatable):
    SUPPORTED_API_TYPES: ClassVar[Tuple[APIType, ...]] = (APIType.QIANFAN,)

    @classmethod
    def create(
        cls,
        messages: List[dict],
        *,
        headers: Optional[HeadersType] = None,
        request_timeout: Optional[float] = None,
        _config_: Optional[ConfigDictType] = None,
    ) -> EBResponse:
        config = _config_ or {}
        resource = cls(**config)
        kwargs = filter_args(
            messages=messages,
        )
        if headers is not None:
            kwargs["headers"] = headers
        if request_timeout is not None:
            kwargs["request_timeout"] = request_timeout
        resp = resource.create_resource(**kwargs)
        return resp

    @classmethod
    async def acreate(
        cls,
        messages: List[dict],
        *,
        headers: Optional[HeadersType] = None,
        request_timeout: Optional[float] = None,
        _config_: Optional[ConfigDictType] = None,
    ) -> EBResponse:
        config = _config_ or {}
        resource = cls(**config)
        kwargs = filter_args(
            messages=messages,
        )
        if headers is not None:
            kwargs["headers"] = headers
        if request_timeout is not None:
            kwargs["request_timeout"] = request_timeout
        resp = await resource.acreate_resource(**kwargs)
        return resp

    def _prepare_create(self, kwargs: Dict[str, Any]) -> Request:
        valid_keys = {
            "messages",
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

        # path
        assert self.SUPPORTED_API_TYPES == (APIType.QIANFAN,)
        if self.api_type is APIType.QIANFAN:
            path = "/chat/chatfile_adv"
        else:
            raise errors.UnsupportedAPITypeError(
                f"Supported API types: {self.get_supported_api_type_names()}"
            )

        # params
        params = {}
        params["messages"] = messages

        # headers
        headers = kwargs.get("headers", None)

        # request_timeout
        request_timeout = kwargs.get("request_timeout", None)

        return Request(
            path=path,
            params=params,
            headers=headers,
            timeout=request_timeout,
        )
