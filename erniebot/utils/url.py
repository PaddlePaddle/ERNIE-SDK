# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

from typing import (List, Tuple)
from urllib.parse import urlencode, urlsplit, urlunsplit

__all__ = ['add_query_params', 'extract_base_url']


def add_query_params(url: str, params: List[Tuple[str, str]]) -> str:
    if len(params) == 0:
        return url

    query = urlencode(params)

    scheme, netloc, path, base_query, fragment = urlsplit(url)

    if base_query:
        query = f"{base_query}&{query}"

    return urlunsplit((scheme, netloc, path, query, fragment))


def extract_base_url(url: str) -> str:
    scheme, netloc, _, _, _ = urlsplit(url)
    return urlunsplit((scheme, netloc, '', '', ''))
