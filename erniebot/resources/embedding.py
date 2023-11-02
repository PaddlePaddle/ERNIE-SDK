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


class Embedding(EBResource, Creatable):
    """Get the embeddings of a given text input."""

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
        user_id: Optional[str] = None,
        _config_: Optional[ConfigDictType] = None,
        headers: Optional[HeadersType] = None,
        request_timeout: Optional[float] = None,
    ) -> EBResponse:
        config = _config_ or {}
        resource = cls(**config)
        resp = resource.create_resource(
            **filter_args(
                model=model,
                input=input,
                user_id=user_id,
                headers=headers,
                request_timeout=request_timeout,
            )
        )
        return resp

    @classmethod
    async def acreate(
        cls,
        model: str,
        input: List[str],
        *,
        user_id: Optional[str] = None,
        _config_: Optional[ConfigDictType] = None,
        headers: Optional[HeadersType] = None,
        request_timeout: Optional[float] = None,
    ) -> EBResponse:
        config = _config_ or {}
        resource = cls(**config)
        resp = await resource.acreate_resource(
            **filter_args(
                model=model,
                input=input,
                user_id=user_id,
                headers=headers,
                request_timeout=request_timeout,
            )
        )
        return resp

    def _prepare_create(self, kwargs: Dict[str, Any]) -> Request:
        def _set_val_if_key_exists(src: dict, dst: dict, key: str) -> None:
            if key in src:
                dst[key] = src[key]

        VALID_KEYS = {
            "model",
            "input",
            "headers",
            "request_timeout",
        }

        invalid_keys = kwargs.keys() - VALID_KEYS

        if len(invalid_keys) > 0:
            raise errors.InvalidArgumentError(f"Invalid keys found in `kwargs`: {list(invalid_keys)}")

        # model
        if "model" not in kwargs:
            raise errors.ArgumentNotFoundError("`model` is not found.")
        model = kwargs["model"]

        # input
        if "input" not in kwargs:
            raise errors.ArgumentNotFoundError("`input` is not found.")
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
        headers = kwargs.get("headers", None)

        # request_timeout
        request_timeout = kwargs.get("request_timeout", None)

        return Request(
            path=path,
            params=params,
            headers=headers,
            timeout=request_timeout,
        )

    def _postprocess_create(self, resp: EBResponse) -> EBResponse:
        return EmbeddingResponse.from_mapping(resp)


class EmbeddingResponse(EBResponse):
    def get_result(self) -> Any:
        embeddings = []
        for res in self.data:
            embeddings.append(res["embedding"])
        return embeddings
