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

import erniebot.errors as errors
from erniebot.api_types import APIType
from erniebot.response import EBResponse
from erniebot.types import FilesType, HeadersType, ParamsType

from .base import EBBackend


class CustomBackend(EBBackend):
    """
    Custom backend for debugging purposes.
    """

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
        files: Optional[FilesType] = None,
        request_timeout: Optional[float] = None,
    ) -> Union[EBResponse, Iterator[EBResponse]]:
        url, headers, data = self._client.prepare_request(
            method,
            path,
            supplied_headers=headers,
            params=params,
            files=files,
        )

        return self._client.send_request(
            method,
            url,
            stream,
            data=data,
            headers=headers,
            files=files,
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
        files: Optional[FilesType] = None,
        request_timeout: Optional[float] = None,
    ) -> Union[EBResponse, AsyncIterator[EBResponse]]:
        url, headers, data = self._client.prepare_request(
            method,
            path,
            supplied_headers=headers,
            params=params,
            files=files,
        )

        return await self._client.asend_request(
            method,
            url,
            stream,
            data=data,
            headers=headers,
            files=files,
            request_timeout=request_timeout,
        )

    def handle_response(self, resp: EBResponse) -> EBResponse:
        if "error_code" in resp and "error_msg" in resp:
            ecode = resp["error_code"]
            emsg = resp["error_msg"]
            if ecode == 17:
                raise errors.RequestLimitError(emsg, ecode=ecode)
            elif ecode == 18:
                raise errors.RateLimitError(emsg, ecode=ecode)
            elif ecode == 110:
                raise errors.InvalidTokenError(emsg, ecode=ecode)
            elif ecode == 111:
                raise errors.TokenExpiredError(emsg, ecode=ecode)
            elif ecode in (336002, 336003, 336006, 336007, 336102):
                raise errors.BadRequestError(emsg, ecode=ecode)
            elif ecode == 336100:
                raise errors.TryAgain(emsg, ecode=ecode)
            else:
                raise errors.APIError(emsg, ecode=ecode)
        else:
            return resp
