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

from typing import Any, AsyncIterator, ClassVar, Dict, Iterator, Optional, Union

from erniebot.api_types import APIType
from erniebot.backends.bce import QianfanLegacyBackend
from erniebot.response import EBResponse
from erniebot.types import HeadersType, ParamsType

from .base import EBBackend


class CustomBackend(EBBackend):
    """Custom backend for debugging purposes."""

    api_type: ClassVar[APIType] = APIType.CUSTOM

    def __init__(self, config_dict: Dict[str, Any]) -> None:
        super().__init__(config_dict=config_dict)

    def request(
        self,
        method: str,
        path: str,
        stream: bool,
        *,
        params: Optional[ParamsType] = None,
        headers: Optional[HeadersType] = None,
        request_timeout: Optional[float] = None,
    ) -> Union[EBResponse, Iterator[EBResponse]]:
        url, headers, data = self._client.prepare_request(
            method,
            path,
            supplied_headers=headers,
            params=params,
        )

        return self._client.send_request(
            method,
            url,
            stream,
            data=data,
            headers=headers,
            request_timeout=request_timeout,
        )

    async def arequest(
        self,
        method: str,
        path: str,
        stream: bool,
        *,
        params: Optional[ParamsType] = None,
        headers: Optional[HeadersType] = None,
        request_timeout: Optional[float] = None,
    ) -> Union[EBResponse, AsyncIterator[EBResponse]]:
        url, headers, data = self._client.prepare_request(
            method,
            path,
            supplied_headers=headers,
            params=params,
        )

        return await self._client.asend_request(
            method,
            url,
            stream,
            data=data,
            headers=headers,
            request_timeout=request_timeout,
        )

    @classmethod
    def handle_response(cls, resp: EBResponse) -> EBResponse:
        return QianfanLegacyBackend.handle_response(resp)
