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

from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import erniebot.errors as errors
from erniebot.api_types import APIType
from erniebot.response import EBResponse
from erniebot.types import ConfigDictType, HeadersType, Request
from erniebot.utils.misc import NOT_GIVEN, NotGiven, filter_args

from .abc import Creatable
from .resource import EBResource

__all__ = ["Embedding", "EmbeddingResponse"]


class Embedding(EBResource, Creatable):
    SUPPORTED_API_TYPES: ClassVar[Tuple[APIType, ...]] = (
        APIType.QIANFAN,
        APIType.AISTUDIO,
    )
    _API_INFO_DICT: ClassVar[Dict[APIType, Dict[str, Any]]] = {
        APIType.QIANFAN: {
            "resource_id": "embeddings",
            "models": {
                "ernie-text-embedding": {
                    "model_id": "embedding-v1",
                },
            },
        },
        APIType.AISTUDIO: {
            "resource_id": "embeddings",
            "models": {
                "ernie-text-embedding": {
                    "model_id": "embedding-v1",
                },
            },
        },
    }

    @classmethod
    def create(
        cls,
        model: str,
        input: List[str],
        *,
        user_id: Union[str, NotGiven] = NOT_GIVEN,
        headers: Optional[HeadersType] = None,
        request_timeout: Optional[float] = None,
        _config_: Optional[ConfigDictType] = None,
    ) -> "EmbeddingResponse":
        """Creates embeddings for the given input texts.

        Args:
            model: Name of the model to use.
            input: Input texts to embed.
            user_id: ID for the end user.
            headers: Custom headers to send with the request.
            request_timeout: Timeout for a single request.
            _config_: Overrides the global settings.

        Returns:
            Response containing the embeddings.
        """
        config = _config_ or {}
        resource = cls(**config)
        kwargs = filter_args(
            model=model,
            input=input,
            user_id=user_id,
        )
        if headers is not None:
            kwargs["headers"] = headers
        if request_timeout is not None:
            kwargs["request_timeout"] = request_timeout
        resp = resource.create_resource(**kwargs)
        return EmbeddingResponse.from_mapping(resp)

    @classmethod
    async def acreate(
        cls,
        model: str,
        input: List[str],
        *,
        user_id: Union[str, NotGiven] = NOT_GIVEN,
        headers: Optional[HeadersType] = None,
        request_timeout: Optional[float] = None,
        _config_: Optional[ConfigDictType] = None,
    ) -> "EmbeddingResponse":
        """Creates embeddings for the given input texts.

        Args:
            model: Name of the model to use.
            input: Input texts to embed.
            user_id: ID for the end user.
            headers: Custom headers to send with the request.
            request_timeout: Timeout for a single request.
            _config_: Overrides the global settings.

        Returns:
            Response containing the embeddings.
        """
        config = _config_ or {}
        resource = cls(**config)
        kwargs = filter_args(
            model=model,
            input=input,
            user_id=user_id,
        )
        if headers is not None:
            kwargs["headers"] = headers
        if request_timeout is not None:
            kwargs["request_timeout"] = request_timeout
        resp = await resource.acreate_resource(**kwargs)
        return EmbeddingResponse.from_mapping(resp)

    def _prepare_create(self, kwargs: Dict[str, Any]) -> Request:
        def _set_val_if_key_exists(src: dict, dst: dict, key: str) -> None:
            if key in src:
                dst[key] = src[key]

        valid_keys = {
            "model",
            "input",
            "headers",
            "request_timeout",
        }

        invalid_keys = kwargs.keys() - valid_keys

        if len(invalid_keys) > 0:
            raise ValueError(f"Invalid keys found in `kwargs`: {list(invalid_keys)}")

        # model
        if "model" not in kwargs:
            raise errors.ArgumentNotFoundError("model")
        model = kwargs["model"]

        # input
        if "input" not in kwargs:
            raise errors.ArgumentNotFoundError("input")
        input = kwargs["input"]

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
        params["input"] = input
        _set_val_if_key_exists(kwargs, params, "input")
        if self.api_type is not APIType.AISTUDIO:
            # The AI Studio backend automatically injects `user_id`.
            _set_val_if_key_exists(kwargs, params, "user_id")

        # headers
        headers: HeadersType = {}
        if self.api_type is APIType.AISTUDIO:
            headers["Content-Type"] = "application/json"
        if "headers" in kwargs:
            headers.update(kwargs["headers"])

        # request_timeout
        request_timeout = kwargs.get("request_timeout", None)

        return Request(
            method="POST",
            path=path,
            params=params,
            headers=headers,
            timeout=request_timeout,
        )


class EmbeddingResponse(EBResponse):
    def get_result(self) -> Any:
        embeddings = []
        for res in self.data:
            embeddings.append(res["embedding"])
        return embeddings
