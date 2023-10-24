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

from typing import (Any, ClassVar, Dict, Optional, Tuple)

import erniebot.errors as errors
from erniebot.api_types import APIType
from erniebot.types import (FilesType, HeadersType, ParamsType, ResponseT)
from erniebot.utils.misc import transform
from .abc import Creatable
from .chat_completion import ChatResponse
from .resource import EBResource


class ChatFile(EBResource, Creatable):
    """Chat with the model about the content of a given file."""

    SUPPORTED_API_TYPES: ClassVar[Tuple[APIType, ...]] = (APIType.QIANFAN, )

    def _prepare_create(self,
                        kwargs: Dict[str, Any]) -> Tuple[str,
                                                         Optional[ParamsType],
                                                         Optional[HeadersType],
                                                         Optional[FilesType],
                                                         bool,
                                                         Optional[float],
                                                         ]:
        VALID_KEYS = {'messages', 'headers', 'request_timeout'}

        invalid_keys = kwargs.keys() - VALID_KEYS

        if len(invalid_keys) > 0:
            raise errors.InvalidArgumentError(
                f"Invalid keys found in `kwargs`: {list(invalid_keys)}")

        # messages
        if 'messages' not in kwargs:
            raise errors.ArgumentNotFoundError("`messages` is not found.")
        messages = kwargs['messages']

        # url
        assert self.SUPPORTED_API_TYPES == (APIType.QIANFAN, )
        if self.api_type is APIType.QIANFAN:
            url = "/chat/chatfile_adv"
        else:
            raise errors.UnsupportedAPITypeError(
                f"Supported API types: {self.get_supported_api_type_names()}")

        # params
        params = {}
        params['messages'] = messages

        # headers
        headers = kwargs.get('headers', None)

        # files
        files = None

        # stream
        stream = False

        # request_timeout
        request_timeout = kwargs.get('request_timeout', None)

        return url, params, headers, files, stream, request_timeout

    def _postprocess_create(self, resp: ResponseT) -> ResponseT:
        return transform(ChatResponse.from_mapping, resp)
