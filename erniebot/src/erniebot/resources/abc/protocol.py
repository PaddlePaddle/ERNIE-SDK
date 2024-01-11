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

from typing import (
    AsyncIterator,
    Iterator,
    Literal,
    Optional,
    Protocol,
    Union,
    overload,
    runtime_checkable,
)

from erniebot.response import EBResponse
from erniebot.types import HeadersType, ParamsType


@runtime_checkable
class Resource(Protocol):
    """Resource protocol.

    Any class that implements all methods defined in this protocol is called a
    resource class.
    """

    @overload
    def request(
        self,
        method: str,
        path: str,
        stream: Literal[False],
        *,
        params: Optional[ParamsType] = ...,
        headers: Optional[HeadersType] = ...,
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
        request_timeout: Optional[float] = ...,
    ) -> Union[EBResponse, Iterator[EBResponse]]:
        ...

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
        """Makes an HTTP request for the resource.

        Args:
            method: HTTP method to use.
            path: Path to request.
            stream: Whether to enable streaming.
            params: Parameters to send.
            headers: Headers to add to the request.
            request_timeout: Request timeout in seconds.

        Returns:
            If `stream` is True, returns an iterator that yields response
            objects. Otherwise returns a response object.
        """
        ...

    @overload
    async def arequest(
        self,
        method: str,
        path: str,
        stream: Literal[False],
        *,
        params: Optional[ParamsType] = ...,
        headers: Optional[HeadersType] = ...,
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
        request_timeout: Optional[float] = ...,
    ) -> Union[EBResponse, AsyncIterator[EBResponse]]:
        ...

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
        """Asynchronous version of `request`."""
        ...
