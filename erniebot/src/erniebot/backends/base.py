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

from typing import AsyncIterator, ClassVar, Iterator, Optional, Union

from erniebot.api_types import APIType
from erniebot.http_client import EBClient
from erniebot.response import EBResponse
from erniebot.types import ConfigDictType, HeadersType, ParamsType


class EBBackend(object):
    api_type: ClassVar[APIType]
    base_url: ClassVar[str]

    def __init__(self, config_dict: ConfigDictType) -> None:
        super().__init__()
        self._base_url = config_dict.get("api_base_url", None) or type(self).base_url
        self._cfg = config_dict
        self._client = EBClient(
            self._base_url,
            session=self._cfg.get("requests_session", None),
            asession=self._cfg.get("aiohttp_session", None),
            response_handler=self.handle_response,
            proxy=self._cfg.get("proxy", None),
        )

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
        raise NotImplementedError

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
        raise NotImplementedError

    @classmethod
    def handle_response(cls, resp: EBResponse) -> EBResponse:
        raise NotImplementedError
