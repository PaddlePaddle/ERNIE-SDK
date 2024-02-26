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

# Modified from
# https://github.com/openai/openai-python/blob/release-v0.28.1/openai/api_requestor.py
# Original LICENSE:
# The MIT License

# Copyright (c) OpenAI (https://openai.com)

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from __future__ import annotations

import asyncio
import http
import json
from contextlib import asynccontextmanager, contextmanager
from json import JSONDecodeError
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Callable,
    Final,
    Generator,
    Iterator,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import aiohttp
import requests

import erniebot

from . import constants, errors
from .response import EBResponse
from .types import HeadersType, ParamsType
from .utils import logging
from .utils.url import add_query_params

__all__ = ["EBClient"]


class EBClient(object):
    """Provides low-level APIs to send HTTP requests and handle responses."""

    DEFAULT_REQUEST_TIMEOUT_SECS: Final[float] = constants.DEFAULT_REQUEST_TIMEOUT_SECS

    _session: Optional[requests.Session]
    _asession: Optional[aiohttp.ClientSession]

    def __init__(
        self,
        base_url: str,
        *,
        session: Optional[requests.Session] = None,
        asession: Optional[aiohttp.ClientSession] = None,
        response_handler: Optional[Callable[[EBResponse], EBResponse]] = None,
        proxy: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._base_url = base_url
        self._session = session
        self._asession = asession
        self._resp_handler = response_handler
        self._proxy = proxy

    def prepare_request(
        self,
        method: str,
        path: str,
        supplied_headers: Optional[HeadersType],
        params: Optional[ParamsType],
    ) -> Tuple[str, HeadersType, Optional[bytes]]:
        url = f"{self._base_url}{path}"
        headers = self._get_request_headers(method, supplied_headers)
        data = None
        method = method.upper()
        if method == "GET" or method == "DELETE":
            if params:
                url = add_query_params(url, [(str(k), str(v)) for k, v in params.items() if v is not None])
        elif method == "POST" or method == "PUT":
            if params:
                data = json.dumps(params).encode()
        else:
            raise errors.ConnectionError(f"Unrecognized HTTP method: {repr(method)}")

        logging.debug("Method: %s", method)
        logging.debug("URL: %s", url)
        logging.debug("Headers: %r", headers)
        logging.debug("Data: %r", data)

        return url, headers, data

    def send_request(
        self,
        method: str,
        url: str,
        stream: bool,
        *,
        data: Optional[bytes] = None,
        headers: Optional[HeadersType] = None,
        request_timeout: Optional[float] = None,
    ) -> Union[EBResponse, Iterator[EBResponse]]:
        ctx = self._make_requests_session_context_manager()
        session = ctx.__enter__()
        should_clean_up_ctx = True

        try:
            result = self.send_request_raw(
                session,
                method.upper(),
                url,
                data=data,
                headers=headers,
                stream=stream,
                request_timeout=request_timeout,
            )
            should_clean_up_result = True
            try:
                resp, got_stream = self._interpret_response(result)
                if stream != got_stream:
                    logging.warning("Unexpected response: %s", resp)
                    logging.warning(
                        f"A {'streamed' if stream else 'non-streamed'} response was expected, "
                        f"but got a {'streamed' if got_stream else 'non-streamed'} response. "
                    )
                if got_stream:

                    def wrap_resp(resp: Iterator) -> Iterator[EBResponse]:
                        try:
                            for r in resp:
                                yield r
                        finally:
                            result.close()
                            ctx.__exit__(None, None, None)

                    assert isinstance(resp, Iterator)
                    resp = wrap_resp(resp)

                    should_clean_up_result = False
                    should_clean_up_ctx = False
            finally:
                if should_clean_up_result:
                    result.close()
        finally:
            if should_clean_up_ctx:
                # We don't care about the exception type and the stack trace.
                ctx.__exit__(None, None, None)

        return resp

    async def asend_request(
        self,
        method: str,
        url: str,
        stream: bool,
        *,
        data: Optional[bytes] = None,
        headers: Optional[HeadersType] = None,
        request_timeout: Optional[float] = None,
    ) -> Union[EBResponse, AsyncIterator[EBResponse]]:
        ctx = self._make_aiohttp_session_context_manager()
        session = await ctx.__aenter__()
        should_clean_up_ctx = True

        try:
            result = await self.asend_request_raw(
                session,
                method.upper(),
                url,
                data=data,
                headers=headers,
                request_timeout=request_timeout,
            )
            should_clean_up_result = True
            try:
                resp, got_stream = await self._interpret_async_response(result)
                if stream != got_stream:
                    logging.warning("Unexpected response: %s", resp)
                    logging.warning(
                        f"A {'streamed' if stream else 'non-streamed'} response was expected, "
                        f"but got a {'streamed' if got_stream else 'non-streamed'} response. "
                    )
                if got_stream:

                    async def wrap_resp(resp: AsyncIterator) -> AsyncIterator[EBResponse]:
                        try:
                            async for r in resp:
                                yield r
                        finally:
                            result.release()
                            await ctx.__aexit__(None, None, None)

                    assert isinstance(resp, AsyncIterator)
                    resp = wrap_resp(resp)

                    should_clean_up_result = False
                    should_clean_up_ctx = False
            finally:
                if should_clean_up_result:
                    result.release()
        finally:
            if should_clean_up_ctx:
                await ctx.__aexit__(None, None, None)

        return resp

    def send_request_raw(
        self,
        session: requests.Session,
        method: str,
        url: str,
        data: Optional[bytes],
        headers: Optional[HeadersType],
        stream: bool,
        request_timeout: Optional[float],
    ) -> requests.Response:
        try:
            result = session.request(
                method,
                url,
                headers=headers,
                data=data,
                stream=stream,
                timeout=request_timeout if request_timeout else self.DEFAULT_REQUEST_TIMEOUT_SECS,
                proxies=session.proxies,
            )
        except requests.exceptions.Timeout as e:
            raise errors.TimeoutError(f"Request timed out: {e}") from e
        except requests.exceptions.RequestException as e:
            raise errors.ConnectionError(f"Error communicating with server: {e}") from e

        return result

    async def asend_request_raw(
        self,
        session: aiohttp.ClientSession,
        method: str,
        url: str,
        data: Optional[bytes],
        headers: Optional[HeadersType],
        request_timeout: Optional[float],
    ) -> aiohttp.ClientResponse:
        timeout = aiohttp.ClientTimeout(
            total=request_timeout if request_timeout else self.DEFAULT_REQUEST_TIMEOUT_SECS
        )

        request_kwargs: dict = {
            "headers": headers,
            "data": data,
            "timeout": timeout,
        }

        try:
            result = await session.request(method=method, url=url, **request_kwargs)
        except (aiohttp.ServerTimeoutError, asyncio.TimeoutError) as e:
            raise errors.TimeoutError(f"Request timed out: {e}") from e
        except aiohttp.ClientError as e:
            raise errors.ConnectionError(f"Error communicating with server: {e}") from e

        return result

    def _get_request_headers(self, method: str, supplied_headers: Optional[HeadersType]) -> HeadersType:
        headers = {}

        headers["User-Agent"] = f"ERNIE-SDK/{erniebot.__version__}"
        # TODO: Add other default headers

        if supplied_headers is not None:
            self._validate_headers(supplied_headers)
            headers.update(supplied_headers)

        return headers

    def _validate_headers(self, supplied_headers: HeadersType) -> None:
        if not isinstance(supplied_headers, dict):
            raise TypeError("`supplied_headers` must be a dictionary.")

        for k, v in supplied_headers.items():
            if not isinstance(k, str):
                raise TypeError("Header keys must be strings.")
            if not isinstance(v, str):
                raise TypeError("Header values must be strings.")

    def _interpret_response(
        self, response: requests.Response
    ) -> Tuple[Union[EBResponse, Iterator[EBResponse]], bool]:
        if "Content-Type" in response.headers and response.headers["Content-Type"].startswith(
            "text/event-stream"
        ):
            return (
                self._interpret_stream_response(response),
                True,
            )
        else:
            return (
                self._interpret_response_line(
                    response.content.decode("utf-8"),
                    response.status_code,
                    response.headers,
                    stream=False,
                ),
                False,
            )

    async def _interpret_async_response(
        self, response: aiohttp.ClientResponse
    ) -> Tuple[Union[EBResponse, AsyncIterator[EBResponse]], bool]:
        if "Content-Type" in response.headers and response.headers["Content-Type"].startswith(
            "text/event-stream"
        ):
            return (
                self._interpret_async_stream_response(response),
                True,
            )
        else:
            try:
                rbody = await response.read()
            except (aiohttp.ServerTimeoutError, asyncio.TimeoutError) as e:
                raise errors.TimeoutError(f"Request timed out: {str(e)}") from e
            else:
                return (
                    self._interpret_response_line(
                        rbody.decode("utf-8"),
                        response.status,
                        response.headers,
                        stream=False,
                    ),
                    False,
                )

    def _interpret_stream_response(self, response: requests.Response) -> Iterator[EBResponse]:
        for line in self._parse_stream(response.iter_lines()):
            resp = self._interpret_response_line(line, response.status_code, response.headers, stream=True)
            yield resp

    async def _interpret_async_stream_response(
        self, response: aiohttp.ClientResponse
    ) -> AsyncIterator[EBResponse]:
        async for line in self._parse_async_stream(response.content):
            resp = self._interpret_response_line(line, response.status, response.headers, stream=True)
            yield resp

    def _interpret_response_line(
        self,
        rbody: str,
        rcode: int,
        rheaders: Mapping[str, Any],
        stream: bool,
    ) -> EBResponse:
        content_type = rheaders.get("Content-Type", "")
        if content_type.startswith("text/plain"):
            decoded_rbody = rbody
        elif content_type.startswith("application/json") or content_type.startswith("text/event-stream"):
            try:
                decoded_rbody = json.loads(rbody)
            except (JSONDecodeError, UnicodeDecodeError) as e:
                raise errors.HTTPRequestError(
                    "Could not decode the response body.",
                    rcode=rcode,
                    rbody=rbody,
                    rheaders=rheaders,
                ) from e
            if not isinstance(decoded_rbody, (str, dict)):
                raise errors.HTTPRequestError(
                    f"The decoded response body has an unsupported type: {type(decoded_rbody)}",
                    rcode=rcode,
                    rbody=rbody,
                    rheaders=rheaders,
                )
        else:
            raise errors.HTTPRequestError(
                f"Unexpected content type: {content_type}",
                rcode=rcode,
                rbody=rbody,
                rheaders=rheaders,
            )

        logging.debug("Decoded response body: %r", decoded_rbody)

        response = EBResponse(rcode=rcode, rbody=decoded_rbody, rheaders=dict(rheaders))
        if rcode != http.HTTPStatus.OK:
            raise errors.HTTPRequestError(
                f"The status code is not {http.HTTPStatus.OK}.",
                rcode=response.rcode,
                rbody=str(response.rbody),
                rheaders=response.rheaders,
            )

        if self._resp_handler is not None:
            response = self._resp_handler(response)
        return response

    def _parse_stream(self, rbody: Iterator[bytes]) -> Iterator[str]:
        for line in rbody:
            _line = self._parse_line(line)
            if _line is not None:
                yield _line

    async def _parse_async_stream(self, rbody: aiohttp.StreamReader) -> AsyncIterator[str]:
        async for line in rbody:
            _line = self._parse_line(line)
            if _line is not None:
                yield _line

    def _parse_line(self, line: bytes) -> Optional[str]:
        if line:
            if line.startswith(constants.STREAM_RESPONSE_PREFIX):
                line = line[len(constants.STREAM_RESPONSE_PREFIX) :]
                return line.decode("utf-8")
            else:
                # Filter out other lines
                return None
        return None

    @contextmanager
    def _make_requests_session_context_manager(self) -> Generator[requests.Session, None, None]:
        if self._session is not None:
            session = self._session
            should_close_session = False
        else:
            session = requests.Session()
            should_close_session = True
        try:
            if self._proxy is not None:
                proxies = {"http": self._proxy, "https": self._proxy}
                session.proxies = proxies
            yield session
        finally:
            if should_close_session:
                session.close()

    @asynccontextmanager
    async def _make_aiohttp_session_context_manager(
        self,
    ) -> AsyncGenerator[aiohttp.ClientSession, None]:
        # TODO: Support proxies
        if self._asession is not None:
            session = self._asession
            should_close_session = False
        else:
            session = aiohttp.ClientSession()
            should_close_session = True
        try:
            yield session
        finally:
            if should_close_session:
                await session.close()
