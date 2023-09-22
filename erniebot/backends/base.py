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

from typing import (Any, AsyncIterator, ClassVar, Dict, Iterator, Optional,
                    Union)

from erniebot.api_types import APIType
from erniebot.http_client import EBClient
from erniebot.response import EBResponse
from erniebot.types import (FilesType, HeadersType, ParamsType)


class EBBackend(object):
    API_TYPE: ClassVar[APIType]
    BASE_URL: ClassVar[str]

    def __init__(self, config_dict: Dict[str, Any]) -> None:
        super().__init__()

        self.api_type = self.API_TYPE
        self.base_url = config_dict.get('api_base_url', None) or self.BASE_URL

        self._cfg = config_dict
        self._client = EBClient(
            self.handle_response, proxy=self._cfg.get('proxy', None))

    def handle_response(self, resp: EBResponse) -> EBResponse:
        raise NotImplementedError

    def request(
            self,
            method: str,
            url: str,
            stream: bool,
            params: Optional[ParamsType]=None,
            headers: Optional[HeadersType]=None,
            files: Optional[FilesType]=None,
            request_timeout: Optional[float]=None,
    ) -> Union[EBResponse,
               Iterator[EBResponse],
               ]:
        raise NotImplementedError

    async def arequest(
        self,
        method: str,
        url: str,
        stream: bool,
        params: Optional[ParamsType]=None,
        headers: Optional[HeadersType]=None,
        files: Optional[FilesType]=None,
        request_timeout: Optional[float]=None,
    ) -> Union[EBResponse,
               AsyncIterator[EBResponse],
               ]:
        raise NotImplementedError

    def _get_full_url(self, path: str) -> str:
        return f"{self.base_url}{path}"
