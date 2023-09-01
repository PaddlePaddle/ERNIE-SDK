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

import asyncio
import time
from typing import (Any, AsyncIterator, Callable, cast, Dict, Iterator,
                    Optional, Union)

from typing_extensions import final, Self

import erniebot.errors as errors
from erniebot.backends import build_backend, convert_str_to_api_type
from erniebot.client import EBClient
from erniebot.config import GlobalConfig
from erniebot.response import EBResponse
from erniebot.types import (ParamsType, HeadersType, FilesType)
from erniebot.utils import logger


class EBResource(object):
    """Resource class with enhanced features.

    This class implements the resource protocol and provides the following
    additional functionalities:
    1. Support synchronous and asynchronous HTTP polling.
    2. Automatically refresh the access token.
    3. Override the global settings.

    This class can be typically used as a mix-in for another resource class to
    facilitate reuse of concrete implementations. Most methods of this class are
    marked as final (e.g., `request`, `arequest`), while some methods can be
    overridden to change the default behavior (e.g., `_create_config_dict`).

    Attributes:
        cfg (Dict[str, Any]): Dictionary that stores global settings.
        client (erniebot.client.EBClient): Low-level client instance.
    """

    MAX_POLLING_RETRIES: int = 20
    POLLING_INTERVAL: int = 5
    _MAX_TOKEN_UPDATE_RETRIES: int = 3

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()

        self._cfg = self._create_config_dict(**kwargs)

        self.api_type = self._cfg['api_type']
        self.timeout = self._cfg['timeout']

        backend = build_backend(self._cfg, self._cfg['api_type'])
        self._backend = backend
        self._client = EBClient(self._cfg, backend)

    @classmethod
    def new_object(cls, **kwargs: Any) -> Self:
        return cls(**kwargs)

    @final
    def request(
            self,
            method: str,
            url: str,
            *,
            stream: bool,
            params: Optional[ParamsType]=None,
            headers: Optional[HeadersType]=None,
            files: Optional[FilesType]=None,
            request_timeout: Optional[float]=None,
    ) -> Union[EBResponse,
               Iterator[EBResponse],
               ]:
        if self.timeout is None:
            return self._request(
                method=method,
                url=url,
                stream=stream,
                params=params,
                headers=headers,
                files=files,
                request_timeout=request_timeout)
        else:
            st_time = time.time()
            while True:
                try:
                    return self._request(
                        method=method,
                        url=url,
                        stream=stream,
                        params=params,
                        headers=headers,
                        files=files,
                        request_timeout=request_timeout)
                except errors.TryAgain as e:
                    if time.time() > st_time + self.timeout:
                        logger.info("Operation timed out. No more attempts.")
                        raise
                    else:
                        logger.info("Another attempt will be made.")

    @final
    async def arequest(
        self,
        method: str,
        url: str,
        *,
        stream: bool,
        params: Optional[ParamsType]=None,
        headers: Optional[HeadersType]=None,
        files: Optional[FilesType]=None,
        request_timeout: Optional[float]=None,
    ) -> Union[EBResponse,
               AsyncIterator[EBResponse],
               ]:
        if self.timeout is None:
            return await self._arequest(
                method=method,
                url=url,
                stream=stream,
                params=params,
                headers=headers,
                files=files,
                request_timeout=request_timeout)
        else:
            st_time = time.time()
            while True:
                try:
                    return await self._arequest(
                        method=method,
                        url=url,
                        stream=stream,
                        params=params,
                        headers=headers,
                        files=files,
                        request_timeout=request_timeout)
                except errors.TryAgain as e:
                    if time.time() > st_time + self.timeout:
                        logger.info("Operation timed out. No more attempts.")
                        raise
                    else:
                        logger.info("Another attempt will be made.")

    @final
    def poll(
            self,
            until: Callable[[EBResponse], bool],
            method: str,
            url: str,
            *,
            params: Optional[ParamsType]=None,
            headers: Optional[HeadersType]=None,
            request_timeout: Optional[float]=None,
    ) -> EBResponse:
        for _ in range(self.MAX_POLLING_RETRIES):
            resp = self._request(
                method=method,
                url=url,
                stream=False,
                params=params,
                headers=headers,
                files=None,
                request_timeout=request_timeout)
            assert isinstance(resp, EBResponse)
            if until(resp):
                return resp
            logger.info(f"Waiting...")
            time.sleep(self.POLLING_INTERVAL)
        else:
            logger.error(f"Max retries exceeded while polling.")
            raise errors.MaxRetriesExceededError

    @final
    async def apoll(
        self,
        until: Callable[[EBResponse], bool],
        method: str,
        url: str,
        *,
        params: Optional[ParamsType]=None,
        headers: Optional[HeadersType]=None,
        request_timeout: Optional[float]=None,
    ) -> EBResponse:
        for _ in range(self.MAX_POLLING_RETRIES):
            resp = await self._arequest(
                method=method,
                url=url,
                stream=False,
                params=params,
                headers=headers,
                files=None,
                request_timeout=request_timeout)
            assert isinstance(resp, EBResponse)
            if until(resp):
                return resp
            logger.info(f"Waiting...")
            await asyncio.sleep(self.POLLING_INTERVAL)
        else:
            logger.error(f"Max retries exceeded while polling.")
            raise errors.MaxRetriesExceededError

    @final
    def _request(
            self,
            method: str,
            url: str,
            stream: bool,
            params: Optional[ParamsType],
            headers: Optional[HeadersType],
            files: Optional[FilesType],
            request_timeout: Optional[float],
    ) -> Union[EBResponse,
               Iterator[EBResponse],
               ]:
        token = self._backend.get_access_token()
        attempts = 0
        while True:
            try:
                return self._client.request(
                    token,
                    method,
                    url,
                    stream,
                    params=params,
                    headers=headers,
                    files=files,
                    request_timeout=request_timeout)
            except (errors.TokenExpiredError, errors.InvalidTokenError):
                attempts += 1
                if attempts <= self._MAX_TOKEN_UPDATE_RETRIES:
                    logger.warning(
                        f"Access token provided is invalid or has expired. An automatic update will be performed before retrying."
                    )
                    token = self._backend.update_access_token()
                    continue
                else:
                    raise

    @final
    async def _arequest(
        self,
        method: str,
        url: str,
        stream: bool,
        params: Optional[ParamsType],
        headers: Optional[HeadersType],
        files: Optional[FilesType],
        request_timeout: Optional[float],
    ) -> Union[EBResponse,
               AsyncIterator[EBResponse],
               ]:
        token = self._backend.get_access_token()
        attempts = 0
        while True:
            try:
                return await self._client.arequest(
                    token,
                    method,
                    url,
                    stream,
                    params=params,
                    headers=headers,
                    files=files,
                    request_timeout=request_timeout)
            except (errors.TokenExpiredError, errors.InvalidTokenError):
                attempts += 1
                if attempts <= self._MAX_TOKEN_UPDATE_RETRIES:
                    logger.warning(
                        f"Access token provided is invalid or has expired. An automatic update will be performed before retrying."
                    )
                    loop = asyncio.get_running_loop()
                    token = await loop.run_in_executor(
                        None, self._backend.update_access_token)
                    continue
                else:
                    raise

    def _create_config_dict(self, **overrides: Any) -> Dict[str, Any]:
        cfg_dict = cast(Dict[str, Any], GlobalConfig().create_dict(**overrides))
        api_type_str = cfg_dict['api_type']
        if not isinstance(api_type_str, str):
            raise TypeError
        api_type = convert_str_to_api_type(api_type_str)
        cfg_dict['api_type'] = api_type
        return cfg_dict
