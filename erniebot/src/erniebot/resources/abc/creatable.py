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

import abc
from typing import Any, AsyncIterator, Dict, Iterator, Union, cast

from erniebot.response import EBResponse
from erniebot.types import Request, RequestWithStream, ResponseT

from .protocol import Resource


class Creatable(Resource):
    """Creatable resource."""

    def create_resource(self, **create_kwargs: Any) -> EBResponse:
        """Creates a resource."""
        req = self._prepare_create(create_kwargs)
        resp = self.request(
            method=req.method,
            path=req.path,
            stream=False,
            params=req.params,
            headers=req.headers,
            request_timeout=req.timeout,
        )
        resp = self._postprocess_create(resp)
        return resp

    async def acreate_resource(self, **create_kwargs: Any) -> EBResponse:
        """Asynchronous version of `create_resource`."""
        req = self._prepare_create(create_kwargs)
        resp = await self.arequest(
            method=req.method,
            path=req.path,
            stream=False,
            params=req.params,
            headers=req.headers,
            request_timeout=req.timeout,
        )
        resp = self._postprocess_create(resp)
        return resp

    @abc.abstractmethod
    def _prepare_create(self, kwargs: Dict[str, Any]) -> Request:
        ...

    def _postprocess_create(self, resp: EBResponse) -> EBResponse:
        return resp


class CreatableWithStreaming(Resource):
    def create_resource(self, **create_kwargs: Any) -> Union[EBResponse, Iterator[EBResponse]]:
        """Creates a resource."""
        req = self._prepare_create(create_kwargs)
        resp = self.request(
            method=req.method,
            path=req.path,
            stream=req.stream,
            params=req.params,
            headers=req.headers,
            request_timeout=req.timeout,
        )
        if isinstance(resp, EBResponse):
            resp = self._postprocess_create(resp)
        else:
            resp = self._postprocess_create(resp)
        return resp

    async def acreate_resource(self, **create_kwargs: Any) -> Union[EBResponse, AsyncIterator[EBResponse]]:
        """Asynchronous version of `create_resource`."""
        req = self._prepare_create(create_kwargs)
        resp = await self.arequest(
            method=req.method,
            path=req.path,
            stream=req.stream,
            params=req.params,
            headers=req.headers,
            request_timeout=req.timeout,
        )
        if isinstance(resp, EBResponse):
            resp = self._postprocess_create(resp)
        else:
            # See https://github.com/python/mypy/issues/16590
            resp = cast(AsyncIterator[EBResponse], resp)
            resp = self._postprocess_create(resp)
        return resp

    @abc.abstractmethod
    def _prepare_create(self, kwargs: Dict[str, Any]) -> RequestWithStream:
        ...

    def _postprocess_create(self, resp: ResponseT) -> ResponseT:
        return resp
