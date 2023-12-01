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
import operator
import time
from typing import (
    Any,
    AsyncIterator,
    Callable,
    ClassVar,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    final,
    overload,
)

import tenacity

import erniebot.constants as constants
import erniebot.errors as errors
import erniebot.utils.logging as logging
from erniebot.api_types import APIType, convert_str_to_api_type
from erniebot.backends import build_backend
from erniebot.config import GlobalConfig
from erniebot.response import EBResponse
from erniebot.types import ConfigDictType, FilesType, HeadersType, ParamsType


class EBResource(object):
    """Resource class with enhanced features.

    This class implements the resource protocol and provides the following
    additional functionalities:
    1. Synchronous and asynchronous HTTP polling.
    2. Support different backends.
    3. Override the global settings.

    This class can be typically used as a mix-in for another resource class to
    facilitate reuse of concrete implementations.
    """

    POLLING_TIMEOUT_SECS: ClassVar[float] = constants.POLLING_TIMEOUT_SECS
    POLLING_INTERVAL_SECS: ClassVar[float] = constants.POLLING_INTERVAL_SECS
    SUPPORTED_API_TYPES: ClassVar[Tuple[APIType, ...]] = ()
    _BUILD_BACKEND_OPTS_DICT: ClassVar[Dict[APIType, Dict[str, Any]]] = {}

    def __init__(self, **config: Any) -> None:
        object.__init__(self)

        self._cfg = self._create_config_dict(config)

        api_type = self._cfg["api_type"]
        if api_type is None:
            raise RuntimeError("API type is not configured.")
        self.api_type = api_type
        self.timeout = self._cfg["timeout"]
        self.retry_after = (self._cfg["min_retry_delay"] or 0, self._cfg["max_retry_delay"] or 0)

        self._backend = build_backend(
            self.api_type,
            self._cfg,
            **self._BUILD_BACKEND_OPTS_DICT.get(self.api_type, {}),
        )

    @classmethod
    def get_supported_api_type_names(cls) -> List[str]:
        return list(map(operator.attrgetter("name"), cls.SUPPORTED_API_TYPES))

    @overload
    def request(
        self,
        method: str,
        path: str,
        stream: Literal[False],
        *,
        params: Optional[ParamsType] = ...,
        headers: Optional[HeadersType] = ...,
        files: Optional[FilesType] = ...,
        request_timeout: Optional[float] = ...,
    ) -> EBResponse:
        ...

    @overload
    def request(
        self,
        method: str,
        path: str,
        stream: Literal[True],
        *,
        params: Optional[ParamsType] = ...,
        headers: Optional[HeadersType] = ...,
        files: Optional[FilesType] = ...,
        request_timeout: Optional[float] = ...,
    ) -> Iterator[EBResponse]:
        ...

    @overload
    def request(
        self,
        method: str,
        path: str,
        stream: bool,
        *,
        params: Optional[ParamsType] = ...,
        headers: Optional[HeadersType] = ...,
        files: Optional[FilesType] = ...,
        request_timeout: Optional[float] = ...,
    ) -> Union[EBResponse, Iterator[EBResponse]]:
        ...

    @final
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
        if self.timeout is None:
            return self._request(
                method=method,
                path=path,
                stream=stream,
                params=params,
                headers=headers,
                files=files,
                request_timeout=request_timeout,
            )
        else:
            retrying = tenacity.Retrying(
                stop=tenacity.stop_after_delay(self.timeout),
                wait=tenacity.wait_exponential(
                    multiplier=1, max=self.retry_after[1], min=self.retry_after[0]
                )
                + tenacity.wait_random(min=0, max=0.5),
                retry=(
                    tenacity.retry_if_exception_type(errors.TryAgain)
                    | tenacity.retry_if_exception_type(errors.RateLimitError)
                    | tenacity.retry_if_exception_type(errors.TimeoutError)
                ),
                before_sleep=lambda retry_state: logging.warning(
                    "Retrying requests: Attempt %s ended with: %s",
                    retry_state.attempt_number,
                    retry_state.outcome,
                ),
                reraise=True,
            )
            for attempt in retrying:
                with attempt:
                    return self._request(
                        method=method,
                        path=path,
                        stream=stream,
                        params=params,
                        headers=headers,
                        files=files,
                        request_timeout=request_timeout,
                    )
            raise AssertionError

    @overload
    async def arequest(
        self,
        method: str,
        path: str,
        stream: Literal[False],
        *,
        params: Optional[ParamsType] = ...,
        headers: Optional[HeadersType] = ...,
        files: Optional[FilesType] = ...,
        request_timeout: Optional[float] = ...,
    ) -> EBResponse:
        ...

    @overload
    async def arequest(
        self,
        method: str,
        path: str,
        stream: Literal[True],
        *,
        params: Optional[ParamsType] = ...,
        headers: Optional[HeadersType] = ...,
        files: Optional[FilesType] = ...,
        request_timeout: Optional[float] = ...,
    ) -> AsyncIterator[EBResponse]:
        ...

    @overload
    async def arequest(
        self,
        method: str,
        path: str,
        stream: bool,
        *,
        params: Optional[ParamsType] = ...,
        headers: Optional[HeadersType] = ...,
        files: Optional[FilesType] = ...,
        request_timeout: Optional[float] = ...,
    ) -> Union[EBResponse, AsyncIterator[EBResponse]]:
        ...

    @final
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
        if self.timeout is None:
            return await self._arequest(
                method=method,
                path=path,
                stream=stream,
                params=params,
                headers=headers,
                files=files,
                request_timeout=request_timeout,
            )
        else:
            async_retrying = tenacity.AsyncRetrying(
                stop=tenacity.stop_after_delay(self.timeout),
                wait=tenacity.wait_exponential(
                    multiplier=1, max=self.retry_after[1], min=self.retry_after[0]
                )
                + tenacity.wait_random(min=0, max=0.5),
                retry=(
                    tenacity.retry_if_exception_type(errors.TryAgain)
                    | tenacity.retry_if_exception_type(errors.RateLimitError)
                    | tenacity.retry_if_exception_type(errors.TimeoutError)
                ),
                before_sleep=lambda retry_state: logging.warning(
                    "Retrying requests: Attempt %s ended with: %s",
                    retry_state.attempt_number,
                    retry_state.outcome,
                ),
                reraise=True,
            )
            async for attempt in async_retrying:
                with attempt:
                    return await self._arequest(
                        method=method,
                        path=path,
                        stream=stream,
                        params=params,
                        headers=headers,
                        files=files,
                        request_timeout=request_timeout,
                    )
            raise AssertionError

    @final
    def poll(
        self,
        until: Callable[[EBResponse], bool],
        method: str,
        path: str,
        *,
        params: Optional[ParamsType] = None,
        headers: Optional[HeadersType] = None,
        request_timeout: Optional[float] = None,
    ) -> EBResponse:
        st_time = time.time()
        while True:
            resp = self.request(
                method=method,
                path=path,
                stream=False,
                params=params,
                headers=headers,
                files=None,
                request_timeout=request_timeout,
            )
            if until(resp):
                return resp
            if time.time() > st_time + self.POLLING_TIMEOUT_SECS:
                raise errors.TimeoutError
            logging.info("Waiting...")
            time.sleep(self.POLLING_INTERVAL_SECS)

    @final
    async def apoll(
        self,
        until: Callable[[EBResponse], bool],
        method: str,
        path: str,
        *,
        params: Optional[ParamsType] = None,
        headers: Optional[HeadersType] = None,
        request_timeout: Optional[float] = None,
    ) -> EBResponse:
        st_time = time.time()
        while True:
            resp = await self.arequest(
                method=method,
                path=path,
                stream=False,
                params=params,
                headers=headers,
                files=None,
                request_timeout=request_timeout,
            )
            if until(resp):
                return resp
            if time.time() > st_time + self.POLLING_TIMEOUT_SECS:
                raise errors.TimeoutError
            logging.info("Waiting...")
            await asyncio.sleep(self.POLLING_INTERVAL_SECS)

    @overload
    def _request(
        self,
        method: str,
        path: str,
        stream: Literal[False],
        params: Optional[ParamsType],
        headers: Optional[HeadersType],
        files: Optional[FilesType],
        request_timeout: Optional[float],
    ) -> EBResponse:
        ...

    @overload
    def _request(
        self,
        method: str,
        path: str,
        stream: Literal[True],
        params: Optional[ParamsType],
        headers: Optional[HeadersType],
        files: Optional[FilesType],
        request_timeout: Optional[float],
    ) -> Iterator[EBResponse]:
        ...

    @overload
    def _request(
        self,
        method: str,
        path: str,
        stream: bool,
        params: Optional[ParamsType],
        headers: Optional[HeadersType],
        files: Optional[FilesType],
        request_timeout: Optional[float],
    ) -> Union[EBResponse, Iterator[EBResponse]]:
        ...

    @final
    def _request(
        self,
        method: str,
        path: str,
        stream: bool,
        params: Optional[ParamsType],
        headers: Optional[HeadersType],
        files: Optional[FilesType],
        request_timeout: Optional[float],
    ) -> Union[EBResponse, Iterator[EBResponse]]:
        resp = self._backend.request(
            method,
            path,
            stream,
            params=params,
            headers=headers,
            files=files,
            request_timeout=request_timeout,
        )
        if stream:
            if not isinstance(resp, Iterator):
                raise TypeError("Expected an iterator of response objects")
            else:
                return resp
        else:
            if not isinstance(resp, EBResponse):
                raise TypeError("Expected a response object")
            else:
                return resp

    @overload
    async def _arequest(
        self,
        method: str,
        path: str,
        stream: Literal[False],
        params: Optional[ParamsType],
        headers: Optional[HeadersType],
        files: Optional[FilesType],
        request_timeout: Optional[float],
    ) -> EBResponse:
        ...

    @overload
    async def _arequest(
        self,
        method: str,
        path: str,
        stream: Literal[True],
        params: Optional[ParamsType],
        headers: Optional[HeadersType],
        files: Optional[FilesType],
        request_timeout: Optional[float],
    ) -> AsyncIterator[EBResponse]:
        ...

    @overload
    async def _arequest(
        self,
        method: str,
        path: str,
        stream: bool,
        params: Optional[ParamsType],
        headers: Optional[HeadersType],
        files: Optional[FilesType],
        request_timeout: Optional[float],
    ) -> Union[EBResponse, AsyncIterator[EBResponse]]:
        ...

    @final
    async def _arequest(
        self,
        method: str,
        path: str,
        stream: bool,
        params: Optional[ParamsType],
        headers: Optional[HeadersType],
        files: Optional[FilesType],
        request_timeout: Optional[float],
    ) -> Union[EBResponse, AsyncIterator[EBResponse]]:
        resp = await self._backend.arequest(
            method,
            path,
            stream,
            params=params,
            headers=headers,
            files=files,
            request_timeout=request_timeout,
        )
        if stream:
            if not isinstance(resp, AsyncIterator):
                raise TypeError("Expected an iterator of response objects")
            else:
                return resp
        else:
            if not isinstance(resp, EBResponse):
                raise TypeError("Expected a response object")
            else:
                return resp

    def _create_config_dict(self, overrides: Any) -> ConfigDictType:
        cfg_dict = GlobalConfig().create_dict(**overrides)
        api_type_str = cfg_dict["api_type"]
        if not isinstance(api_type_str, str):
            raise TypeError
        api_type = convert_str_to_api_type(api_type_str)
        cfg_dict["api_type"] = api_type
        return cfg_dict
