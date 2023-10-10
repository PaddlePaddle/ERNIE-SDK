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

import os
from typing import (Any, AsyncIterator, ClassVar, Dict, Iterator, Optional,
                    Union)

import erniebot.errors as errors
from erniebot.api_types import APIType
from erniebot.response import EBResponse
from erniebot.types import (FilesType, HeadersType, ParamsType)
from erniebot.utils.logging import logger
from .base import EBBackend


class AIStudioBackend(EBBackend):
    API_TYPE: ClassVar[APIType] = APIType.AISTUDIO
    BASE_URL: ClassVar[str] = "https://aistudio.baidu.com/llm/lmapi/v1"

    def __init__(self, config_dict: Dict[str, Any]) -> None:
        super().__init__(config_dict=config_dict)
        access_token = self._cfg.get('access_token', None)
        if access_token is None:
            access_token = os.environ.get('AISTUDIO_ACCESS_TOKEN', None)
            if access_token is None:
                raise RuntimeError("No access token is configured.")
        self._access_token = access_token

    def handle_response(self, resp: EBResponse) -> EBResponse:
        if resp['errorCode'] != 0:
            ecode = resp['errorCode']
            emsg = resp['errorMsg']
            if ecode == 2:
                raise errors.ServiceUnavailableError(emsg, ecode=ecode)
            elif ecode == 6:
                raise errors.PermissionError(emsg, ecode=ecode)
            elif ecode in (17, 18, 19, 40407):
                raise errors.RequestLimitError(emsg, ecode=ecode)
            elif ecode in (110, 40401):
                raise errors.InvalidTokenError(emsg, ecode=ecode)
            elif ecode == 111:
                raise errors.TokenExpiredError(emsg, ecode=ecode)
            elif ecode == 336003:
                raise errors.InvalidParameterError(emsg, ecode=ecode)
            elif ecode == 336100:
                raise errors.TryAgain(emsg, ecode=ecode)
            else:
                raise errors.APIError(emsg, ecode=ecode)
        else:
            return EBResponse(resp.rcode, resp.result, resp.rheaders)

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
        url = self._get_full_url(url)
        url, headers, data = self._client.prepare_request(
            method,
            url,
            supplied_headers=headers,
            params=params,
            files=files,
        )
        headers = self._add_aistudio_fields_to_headers(headers)
        return self._client.send_request(
            method,
            url,
            stream,
            data=data,
            headers=headers,
            files=files,
            request_timeout=request_timeout,
            base_url=self.base_url,
        )

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
        url = self._get_full_url(url)
        url, headers, data = self._client.prepare_request(
            method,
            url,
            supplied_headers=headers,
            params=params,
            files=files,
        )
        headers = self._add_aistudio_fields_to_headers(headers)
        return await self._client.asend_request(
            method,
            url,
            stream,
            data=data,
            headers=headers,
            files=files,
            request_timeout=request_timeout,
        )

    def _add_aistudio_fields_to_headers(self,
                                        headers: HeadersType) -> HeadersType:
        if 'Authorization' in headers:
            logger.warning(
                "Key 'Authorization' already exists in `headers`: %r",
                headers['Authorization'])
        headers['Authorization'] = f"token {self._access_token}"
        return headers
