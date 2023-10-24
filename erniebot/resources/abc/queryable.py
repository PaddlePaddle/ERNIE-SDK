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
from typing import (Any, Dict, Optional, Tuple)

from erniebot.response import EBResponse
from erniebot.types import (ParamsType, HeadersType)
from .protocol import Resource


class Queryable(Resource):
    """Queryable resource."""

    @classmethod
    def query(cls, **kwargs: Any) -> EBResponse:
        """Query a resource."""
        config = kwargs.pop('_config_', {})
        resource = cls.new_object(**config)
        query_kwargs = kwargs
        return resource.query_resource(**query_kwargs)

    @classmethod
    async def aquery(cls, **kwargs: Any) -> EBResponse:
        """Asynchronous version of `query`."""
        config = kwargs.pop('_config_', {})
        resource = cls.new_object(**config)
        query_kwargs = kwargs
        resp = await resource.aquery_resource(**query_kwargs)
        return resp

    def query_resource(self, **query_kwargs: Any) -> EBResponse:
        url, params, headers, request_timeout = self._prepare_query(
            query_kwargs)
        resp = self.request(
            method='POST',
            url=url,
            params=params,
            stream=False,
            headers=headers,
            request_timeout=request_timeout)
        resp = self._postprocess_query(resp)
        return resp

    async def aquery_resource(self, **query_kwargs: Any) -> EBResponse:
        url, params, headers, request_timeout = self._prepare_query(
            query_kwargs)
        resp = await self.arequest(
            method='POST',
            url=url,
            params=params,
            stream=False,
            headers=headers,
            request_timeout=request_timeout)
        resp = self._postprocess_query(resp)
        return resp

    @abc.abstractmethod
    def _prepare_query(self,
                       kwargs: Dict[str, Any]) -> Tuple[str,
                                                        Optional[ParamsType],
                                                        Optional[HeadersType],
                                                        Optional[float],
                                                        ]:
        ...

    def _postprocess_query(self, resp: EBResponse) -> EBResponse:
        return resp
