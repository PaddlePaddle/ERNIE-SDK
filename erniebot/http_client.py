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

# Modified from https://github.com/openai/openai-python/blob/main/openai/api_requestor.py
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
import threading
import time
from contextlib import asynccontextmanager
from json import JSONDecodeError
from typing import (Any, AsyncIterator, Callable, Dict, Iterator, Mapping,
                    Optional, Tuple, Union)

import aiohttp
import requests

from . import errors
from .response import EBResponse
from .types import (FilesType, HeadersType, ParamsType)
from .utils.logging import logger
from .utils.url import add_query_params, extract_base_url

__all__ = ['EBClient']

_thread_context = threading.local()


class EBClient(object):
    """A client class that provides low-level APIs to handle HTTP requests and
    responses."""

    MAX_CONNECTION_RETRIES: int = 2
    MAX_SESSION_LIFETIME_SECS: int = 180
    TIMEOUT_SECS: int = 600

    def __init__(
            self,
            response_handler: Optional[Callable[[EBResponse], EBResponse]]=None,
            proxy: Optional[str]=None) -> None:
        super().__init__()
        self._resp_handler = response_handler
        self._proxy = proxy

    def prepare_request(
            self,
            method: str,
            url: str,
            supplied_headers: Optional[HeadersType],
            params: Optional[ParamsType],
            files: Optional[FilesType],
    ) -> Tuple[str,
               HeadersType,
               Optional[bytes],
               ]:
        headers = self._validate_headers(supplied_headers)

        data = None
        method = method.upper()
        if method == 'GET' or method == 'DELETE':
            if params:
                url = add_query_params(url, [(str(k), str(v))
                                             for k, v in params.items()
                                             if v is not None])
        elif method in {'POST', 'PUT'}:
            if params and not files:
                data = json.dumps(params).encode()
                headers['Content-Type'] = 'application/json'
        else:
            raise errors.ConnectionError(
                f"Unrecognized HTTP method {repr(method)}.")

        headers = self.get_request_headers(method, headers)

        logger.debug("Method: %s", method)
        logger.debug("URL: %s", url)
        logger.debug("Headers: %r", headers)
        logger.debug("Data: %r", data)

        return url, headers, data

    def send_request(
            self,
            method: str,
            url: str,
            stream: bool,
            data: Optional[bytes]=None,
            headers: Optional[HeadersType]=None,
            files: Optional[FilesType]=None,
            request_timeout: Optional[float]=None,
            base_url: Optional[str]=None,
    ) -> Union[EBResponse,
               Iterator[EBResponse],
               ]:
        result = self.send_request_raw(
            method.lower(),
            url,
            base_url if base_url is not None else extract_base_url(url),
            data=data,
            headers=headers,
            files=files,
            stream=stream,
            request_timeout=request_timeout,
        )
        resp, got_stream = self._interpret_response(result, stream)
        if stream != got_stream:
            logger.warning("Unexpected response: %s", resp)
            logger.warning(
                f"A {'streamed' if stream else 'non-streamed'} response was expected, "
                f"but got a {'streamed' if got_stream else 'non-streamed'} response. "
            )
        return resp

    async def asend_request(
        self,
        method: str,
        url: str,
        stream: bool,
        data: Optional[bytes]=None,
        headers: Optional[HeadersType]=None,
        files: Optional[FilesType]=None,
        request_timeout: Optional[float]=None,
    ) -> Union[EBResponse,
               AsyncIterator[EBResponse],
               ]:
        # XXX: Should we consider session reuse?
        ctx = self._make_aiohttp_session()
        session = await ctx.__aenter__()
        try:
            result = await self.asend_request_raw(
                method.lower(),
                url,
                session,
                files=files,
                data=data,
                headers=headers,
                request_timeout=request_timeout,
            )
            resp, got_stream = await self._interpret_async_response(result,
                                                                    stream)
            if stream != got_stream:
                logger.warning("Unexpected response: %s", resp)
                logger.warning(
                    f"A {'streamed' if stream else 'non-streamed'} response was expected, "
                    f"but got a {'streamed' if got_stream else 'non-streamed'} response. "
                )
        except Exception as e:
            await ctx.__aexit__(None, None, None)
            raise e
        if isinstance(resp, AsyncIterator):

            async def wrap_resp(
                resp: AsyncIterator) -> AsyncIterator[EBResponse]:
                try:
                    async for r in resp:
                        yield r
                finally:
                    await ctx.__aexit__(None, None, None)

            return wrap_resp(resp)
        else:
            await ctx.__aexit__(None, None, None)
            return resp

    def get_request_headers(self, method: str,
                            extra: HeadersType) -> HeadersType:
        headers = {}

        # TODO: Add User-Agent header and other headers
        headers.update(extra)

        return headers

    def send_request_raw(
            self,
            method: str,
            url: str,
            base_url: str,
            data: Optional[bytes],
            headers: Optional[HeadersType],
            files: Optional[FilesType],
            stream: bool,
            request_timeout: Optional[float],
    ) -> requests.Response:

        if not hasattr(_thread_context, 'sessions'):
            _thread_context.sessions = {}
            _thread_context.session_create_times = {}
        if base_url not in _thread_context.sessions:
            _thread_context.sessions[base_url] = self._make_session()
            _thread_context.session_create_times[base_url] = time.time()
        elif (time.time() - _thread_context.session_create_times[base_url] >=
              self.MAX_SESSION_LIFETIME_SECS):
            _thread_context.sessions[base_url].close()
            _thread_context.sessions[base_url] = self._make_session()
            _thread_context.session_create_times[base_url] = time.time()
        session = _thread_context.sessions[base_url]

        try:
            result = requests.request(
                method,
                url,
                headers=headers,
                data=data,
                files=files,
                stream=stream,
                timeout=request_timeout
                if request_timeout else self.TIMEOUT_SECS,
                proxies=session.proxies,
            )
        except requests.exceptions.Timeout as e:
            raise errors.TimeoutError(f"Request timed out: {e}") from e
        except requests.exceptions.RequestException as e:
            raise errors.ConnectionError(
                f"Error communicating with server: {e}") from e

        logger.debug("API response headers: %r", result.headers)

        return result

    async def asend_request_raw(
        self,
        method: str,
        url: str,
        session: aiohttp.ClientSession,
        data: Optional[bytes],
        headers: Optional[HeadersType],
        files: Optional[FilesType],
        request_timeout: Optional[float],
    ) -> aiohttp.ClientResponse:

        if files is not None:
            raise TypeError("`files` is currently not supported.")

        timeout = aiohttp.ClientTimeout(total=request_timeout if request_timeout
                                        else self.TIMEOUT_SECS)

        request_kwargs: dict = {
            'headers': headers,
            'data': data,
            'timeout': timeout,
        }
        proxy = self._proxy
        if proxy is not None:
            request_kwargs['proxy'] = proxy

        try:
            result = await session.request(
                method=method, url=url, **request_kwargs)
        except (aiohttp.ServerTimeoutError, asyncio.TimeoutError) as e:
            raise errors.TimeoutError(f"Request timed out: {e}") from e
        except aiohttp.ClientError as e:
            raise errors.ConnectionError(
                f"Error communicating with server: {e}") from e

        logger.debug("API response headers: %r", result.headers)

        return result

    def _validate_headers(
            self, supplied_headers: Optional[HeadersType]) -> HeadersType:
        headers: dict = {}

        if supplied_headers is None:
            return headers

        if not isinstance(supplied_headers, dict):
            raise TypeError("`supplied_headers` must be a dictionary.")

        for k, v in supplied_headers.items():
            if not isinstance(k, str):
                raise TypeError("Header keys must be strings.")
            if not isinstance(v, str):
                raise TypeError("Header values must be strings.")
            headers[k] = v

        return headers

    def _parse_line(self, line: bytes) -> Optional[str]:
        if line:
            if line.startswith(b"data: "):
                line = line[len(b"data: "):]
                return line.decode('utf-8')
            else:
                # Filter out other lines
                return None
        return None

    def _parse_stream(self, rbody: Iterator[bytes]) -> Iterator[str]:
        for line in rbody:
            _line = self._parse_line(line)
            if _line is not None:
                yield _line

    async def _parse_stream_async(
        self, rbody: aiohttp.StreamReader) -> AsyncIterator[str]:
        async for line in rbody:
            _line = self._parse_line(line)
            if _line is not None:
                yield _line

    def _interpret_response(
            self, result: requests.Response,
            stream: bool) -> Tuple[Union[EBResponse, Iterator[EBResponse]],
                                   bool,
                                   ]:
        if stream and result.headers['Content-Type'].startswith(
                'text/event-stream'):
            return self._interpret_stream_response(
                result.iter_lines(), result.status_code, result.headers), True
        else:
            return self._interpret_response_line(
                result.content.decode('utf-8'),
                result.status_code,
                result.headers,
                stream=False), False

    async def _interpret_async_response(
        self,
        result: aiohttp.ClientResponse,
        stream: bool,
    ) -> Tuple[Union[EBResponse, AsyncIterator[EBResponse]],
               bool,
               ]:
        if stream and result.headers['Content-Type'].startswith(
                'text/event-stream'):
            return self._interpret_async_stream_response(
                result.content, result.status, result.headers), True
        else:
            try:
                await result.read()
            except (aiohttp.ServerTimeoutError, asyncio.TimeoutError) as e:
                raise errors.TimeoutError(f"Request timed out: {str(e)}") from e
            except aiohttp.ClientError as e:
                logger.warning("Ignoring exception.", exc_info=e)
                logger.warning("API response body: %r", result.content)
            return self._interpret_response_line(
                (await result.read()).decode("utf-8"),
                result.status,
                result.headers,
                stream=False), False

    def _interpret_stream_response(
            self,
            rbody: Iterator[bytes],
            rcode: int,
            rheaders: Mapping[str, Any],
    ) -> Iterator[EBResponse]:
        for line in self._parse_stream(rbody):
            resp = self._interpret_response_line(
                line, rcode, rheaders, stream=True)
            yield resp

    async def _interpret_async_stream_response(
        self,
        rbody: aiohttp.StreamReader,
        rcode: int,
        rheaders: Mapping[str, Any],
    ) -> AsyncIterator[EBResponse]:
        async for line in self._parse_stream_async(rbody):
            resp = self._interpret_response_line(
                line, rcode, rheaders, stream=True)
            yield resp

    def _interpret_response_line(
            self,
            rbody: str,
            rcode: int,
            rheaders: Mapping[str, Any],
            stream: bool,
    ) -> EBResponse:
        content_type = rheaders.get('Content-Type', '')
        if content_type.startswith('text/plain'):
            decoded_rbody = rbody
        elif content_type.startswith(
                'application/json') or content_type.startswith(
                    'text/event-stream'):
            try:
                decoded_rbody = json.loads(rbody)
            except (JSONDecodeError, UnicodeDecodeError) as e:
                raise errors.HTTPRequestError(
                    f"Cannot decode response body.",
                    rcode=rcode,
                    rbody=rbody,
                    rheaders=rheaders) from e
            if not isinstance(decoded_rbody, (str, dict)):
                raise errors.HTTPRequestError(
                    f"Decoded response body has an unsupported type: {type(decoded_rbody)}",
                    rcode=rcode,
                    rbody=rbody,
                    rheaders=rheaders)
        else:
            raise errors.HTTPRequestError(
                f"Unexpected content type: {repr(content_type)}",
                rcode=rcode,
                rbody=rbody,
                rheaders=rheaders)

        resp = EBResponse(
            rcode=rcode, rbody=decoded_rbody, rheaders=dict(rheaders))

        if rcode != http.HTTPStatus.OK:
            raise errors.HTTPRequestError(
                f"Status code is not {http.HTTPStatus.OK}.",
                rcode=resp.rcode,
                rbody=str(resp.rbody),
                rheaders=resp.rheaders)

        if self._resp_handler is not None:
            resp = self._resp_handler(resp)
        return resp

    def _make_session(self) -> requests.Session:
        s = requests.Session()
        proxies = self._get_proxies(self._proxy)
        if proxies:
            logger.debug("Use proxies: %r", proxies)
            s.proxies = proxies
        s.mount(
            'https://',
            requests.adapters.HTTPAdapter(
                max_retries=self.MAX_CONNECTION_RETRIES))
        return s

    @staticmethod
    def _get_proxies(proxy: Optional[str]) -> Optional[Dict[str, str]]:
        if proxy is None:
            return None
        else:
            return {'http': proxy, 'https': proxy}

    @staticmethod
    @asynccontextmanager
    async def _make_aiohttp_session() -> AsyncIterator[aiohttp.ClientSession]:
        async with aiohttp.ClientSession() as session:
            yield session
