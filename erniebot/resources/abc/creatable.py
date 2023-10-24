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
from typing import (Any, AsyncIterator, Dict, Iterator, Optional, Tuple, Union)

from erniebot.response import EBResponse
from erniebot.types import (ParamsType, HeadersType, FilesType, ResponseT)
from .protocol import Resource


class Creatable(Resource):
    """Creatable resource."""

    @classmethod
    def create(cls, **kwargs: Any) -> Union[EBResponse, Iterator[EBResponse]]:
        """Create a resource."""
        config = kwargs.pop('_config_', {})
        resource = cls.new_object(**config)
        create_kwargs = kwargs
        return resource.create_resource(**create_kwargs)

    @classmethod
    async def acreate(
        cls, **kwargs: Any) -> Union[EBResponse, AsyncIterator[EBResponse]]:
        """Asynchronous version of `create`."""
        config = kwargs.pop('_config_', {})
        resource = cls.new_object(**config)
        create_kwargs = kwargs
        resp = await resource.acreate_resource(**create_kwargs)
        return resp

    def create_resource(
            self,
            **create_kwargs: Any) -> Union[EBResponse, Iterator[EBResponse]]:
        url, params, headers, files, stream, request_timeout = self._prepare_create(
            create_kwargs)
        resp = self.request(
            method='POST',
            url=url,
            stream=stream,
            params=params,
            headers=headers,
            files=files,
            request_timeout=request_timeout)
        # See https://github.com/python/mypy/issues/1533
        resp = self._postprocess_create(resp)  # type: ignore
        return resp

    async def acreate_resource(
        self,
        **create_kwargs: Any) -> Union[EBResponse, AsyncIterator[EBResponse]]:
        url, params, headers, files, stream, request_timeout = self._prepare_create(
            create_kwargs)
        resp = await self.arequest(
            method='POST',
            url=url,
            stream=stream,
            params=params,
            headers=headers,
            files=files,
            request_timeout=request_timeout)
        resp = self._postprocess_create(resp)  # type: ignore
        return resp

    @abc.abstractmethod
    def _prepare_create(self,
                        kwargs: Dict[str, Any]) -> Tuple[str,
                                                         Optional[ParamsType],
                                                         Optional[HeadersType],
                                                         Optional[FilesType],
                                                         bool,
                                                         Optional[float],
                                                         ]:
        ...

    def _postprocess_create(self, resp: ResponseT) -> ResponseT:
        return resp
