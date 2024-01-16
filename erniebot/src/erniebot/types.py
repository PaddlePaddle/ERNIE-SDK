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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Iterator, Optional, TypeVar

from typing_extensions import TypeAlias

from .response import EBResponse

__all__ = [
    "ConfigDictType",
    "HeadersType",
    "ParamsType",
    "Request",
    "RequestWithStream",
    "ResponseT",
]

ConfigDictType: TypeAlias = Dict[str, Optional[Any]]

HeadersType: TypeAlias = Dict[str, str]
ParamsType: TypeAlias = Dict[str, Any]

ResponseT = TypeVar("ResponseT", EBResponse, Iterator[EBResponse], AsyncIterator[EBResponse])


@dataclass
class Request(object):
    method: str
    path: str
    params: ParamsType
    headers: HeadersType
    timeout: Optional[float] = None


@dataclass
class RequestWithStream(Request):
    stream: bool = False
