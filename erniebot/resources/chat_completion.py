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
from erniebot.types import (ParamsType, HeadersType, FilesType)
from .abc import Creatable
from .resource import EBResource


class ChatCompletion(EBResource, Creatable):
    """Given a conversation, get a new reply from the model."""

    SUPPORTED_API_TYPES: ClassVar[Tuple[APIType, ...]] = (APIType.QIANFAN,
                                                          APIType.AISTUDIO)
    _API_INFO_DICT: ClassVar[Dict[APIType, Dict[str, Any]]] = {
        APIType.QIANFAN: {
            'resource_id': 'chat',
            'models': {
                'ernie-bot-3.5': {
                    'model_id': 'completions',
                },
                'ernie-bot-turbo': {
                    'model_id': 'eb-instant',
                },
            },
        },
        APIType.AISTUDIO: {
            'resource_id': 'chat',
            'models': {
                'ernie-bot-3.5': {
                    'model_id': 'completions',
                },
                'ernie-bot-turbo': {
                    'model_id': 'eb-instant',
                },
            },
        },
    }

    def _prepare_create(self,
                        kwargs: Dict[str, Any]) -> Tuple[str,
                                                         Optional[ParamsType],
                                                         Optional[HeadersType],
                                                         Optional[FilesType],
                                                         bool,
                                                         Optional[float],
                                                         ]:
        def _set_val_if_key_exists(src: dict, dst: dict, key: str) -> None:
            if key in src:
                dst[key] = src[key]

        VALID_KEYS = {
            'model', 'messages', 'functions', 'stream', 'temperature', 'top_p',
            'penalty_score', 'user_id', 'headers', 'request_timeout'
        }

        invalid_keys = kwargs.keys() - VALID_KEYS

        if len(invalid_keys) > 0:
            raise errors.InvalidArgumentError(
                f"Invalid keys found in `kwargs`: {list(invalid_keys)}")

        # model
        if 'model' not in kwargs:
            raise errors.ArgumentNotFoundError("`model` is not found.")
        model = kwargs['model']

        # messages
        if 'messages' not in kwargs:
            raise errors.ArgumentNotFoundError("`messages` is not found.")
        messages = kwargs['messages']

        # url
        if self.api_type in self.SUPPORTED_API_TYPES:
            api_info = self._API_INFO_DICT[self.api_type]
            if model not in api_info['models']:
                raise errors.InvalidArgumentError(
                    f"{repr(model)} is not a supported model.")
            url = f"/{api_info['resource_id']}/{api_info['models'][model]['model_id']}"
        else:
            raise errors.UnsupportedAPITypeError(
                f"Supported API types: {self.get_supported_api_type_names()}")

        # params
        params = {}
        params['messages'] = messages
        if 'functions' in kwargs:
            functions = kwargs['functions']
            required_keys = ('name', 'description', 'parameters')
            for idx, function in enumerate(functions):
                missing_keys = [
                    key for key in required_keys if key not in function
                ]
                if len(missing_keys) > 0:
                    raise errors.InvalidArgumentError(
                        f"Function {idx} does not contain required keys: {missing_keys}"
                    )
            params['functions'] = functions
        _set_val_if_key_exists(kwargs, params, 'stream')
        _set_val_if_key_exists(kwargs, params, 'temperature')
        _set_val_if_key_exists(kwargs, params, 'top_p')
        _set_val_if_key_exists(kwargs, params, 'penalty_score')
        if self.api_type is not APIType.AISTUDIO:
            # NOTE: The AISTUDIO backend automatically injects `user_id`.
            _set_val_if_key_exists(kwargs, params, 'user_id')

        # headers
        headers = kwargs.get('headers', None)

        # files
        files = None

        # stream
        stream = kwargs.get('stream', False)

        # request_timeout
        request_timeout = kwargs.get('request_timeout', None)

        return url, params, headers, files, stream, request_timeout
