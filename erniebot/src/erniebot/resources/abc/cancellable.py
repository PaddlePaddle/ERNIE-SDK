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
from typing import Any, Dict

from erniebot.response import EBResponse
from erniebot.types import Request

from .protocol import Resource


class Cancellable(Resource):
    """Cancellable resource."""

    def cancel_resource(self, **cancel_kwargs: Any) -> EBResponse:
        """Cancels a long-running operation."""
        req = self._prepare_cancel(cancel_kwargs)
        resp = self.request(
            method=req.method,
            path=req.path,
            stream=False,
            params=req.params,
            headers=req.headers,
            request_timeout=req.timeout,
        )
        resp = self._postprocess_cancel(resp)
        return resp

    async def acancel_resource(self, **cancel_kwargs: Any) -> EBResponse:
        """Asynchronous version of `cancel_resource`."""
        req = self._prepare_cancel(cancel_kwargs)
        resp = await self.arequest(
            method=req.method,
            path=req.path,
            stream=False,
            params=req.params,
            headers=req.headers,
            request_timeout=req.timeout,
        )
        resp = self._postprocess_cancel(resp)
        return resp

    @abc.abstractmethod
    def _prepare_cancel(self, kwargs: Dict[str, Any]) -> Request:
        ...

    def _postprocess_cancel(self, resp: EBResponse) -> EBResponse:
        return resp
