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

from typing import (Any, ClassVar, Dict, List, Optional, Tuple)

import jsonschema
import jsonschema.exceptions

import erniebot.errors as errors
from erniebot.api_types import APIType
from erniebot.response import EBResponse
from erniebot.types import (FilesType, HeadersType, ParamsType, ResponseT)
from erniebot.utils.misc import transform
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
                'ernie-bot': {
                    'model_id': 'completions',
                },
                'ernie-bot-turbo': {
                    'model_id': 'eb-instant',
                },
                'ernie-bot-4': {
                    'model_id': 'completions_pro',
                },
            },
        },
        APIType.AISTUDIO: {
            'resource_id': 'chat',
            'models': {
                'ernie-bot': {
                    'model_id': 'completions',
                },
                'ernie-bot-turbo': {
                    'model_id': 'eb-instant',
                },
                'ernie-bot-4': {
                    'model_id': 'completions_pro',
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
            'model', 'messages', 'functions', 'temperature', 'top_p',
            'penalty_score', 'system', 'user_id', 'stream', 'headers',
            'request_timeout'
        }

        invalid_keys = kwargs.keys() - VALID_KEYS

        if len(invalid_keys) > 0:
            raise errors.InvalidArgumentError(
                f"Invalid keys found in `kwargs`: {list(invalid_keys)}")

        # model
        if 'model' not in kwargs:
            raise errors.ArgumentNotFoundError("`model` is not found.")
        model = kwargs['model']
        # For backward compatibility
        if model == 'ernie-bot-3.5':
            model = 'ernie-bot'

        # messages
        if 'messages' not in kwargs:
            raise errors.ArgumentNotFoundError("`messages` is not found.")
        messages = kwargs['messages']
        self._validate_messages(messages)

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
            self._validate_functions(functions)
            params['functions'] = functions
        if 'temperature' in kwargs:
            temperature = kwargs['temperature']
            if temperature < 1e-7 or temperature > 1.:
                raise errors.InvalidArgumentError(
                    "`temperature` must be in the range (0, 1].")
            params['temperature'] = temperature
        if 'top_p' in kwargs:
            top_p = kwargs['top_p']
            if top_p < 0. or top_p > 1.:
                raise errors.InvalidArgumentError(
                    "`top_p` must be in the range [0, 1].")
            params['top_p'] = top_p
        if 'penalty_score' in kwargs:
            penalty_score = kwargs['penalty_score']
            if penalty_score < 1. or penalty_score > 2.:
                raise errors.InvalidArgumentError(
                    "`penalty_score` must be in the range [1, 2].")
            params['penalty_score'] = penalty_score
        if 'system' in kwargs:
            system = kwargs['system']
            if len(system) > 1024:
                raise errors.InvalidArgumentError(
                    "`system` must have less than 1024 characters.")
            params['system'] = system
        if self.api_type is not APIType.AISTUDIO:
            # The AI Studio backend automatically injects `user_id`.
            _set_val_if_key_exists(kwargs, params, 'user_id')
        _set_val_if_key_exists(kwargs, params, 'stream')

        # headers
        headers = kwargs.get('headers', None)

        # files
        files = None

        # stream
        stream = kwargs.get('stream', False)

        # request_timeout
        request_timeout = kwargs.get('request_timeout', None)

        return url, params, headers, files, stream, request_timeout

    def _postprocess_create(self, resp: ResponseT) -> ResponseT:
        return transform(ChatResponse.from_mapping, resp)

    @classmethod
    def _validate_messages(cls, messages: List[dict]) -> None:
        if len(messages) % 2 != 1:
            raise errors.InvalidArgumentError(
                "`messages` must have an odd number of elements.")
        for idx, message in enumerate(messages):
            if 'role' not in message:
                raise errors.InvalidArgumentError(
                    f"Message {idx} does not have a role.")
            if 'content' not in message:
                raise errors.InvalidArgumentError(
                    f"Message {idx} has no content.")
            if idx % 2 == 0:
                if message['role'] not in ('user', 'function'):
                    raise errors.InvalidArgumentError(
                        f"Message {idx} has an invalid role: {message['role']}")
            else:
                if message['role'] != 'assistant':
                    raise errors.InvalidArgumentError(
                        f"Message {idx} has an invalid role: {message['role']}")
            if message['role'] == 'function':
                if 'name' not in message:
                    raise errors.InvalidArgumentError(
                        f"Message {idx} does not contain the function name.")

    @classmethod
    def _validate_functions(cls, functions: List[dict]) -> None:
        required_keys = ('name', 'description', 'parameters')
        optional_keys = ('responses', 'examples', 'plugin_id')
        valid_keys = set(required_keys + optional_keys)
        for idx, function in enumerate(functions):
            missing_keys = [key for key in required_keys if key not in function]
            if len(missing_keys) > 0:
                raise errors.InvalidArgumentError(
                    f"Function {idx} does not contain the required keys: {missing_keys}"
                )
            invalid_keys = function.keys() - valid_keys
            if len(invalid_keys) > 0:
                raise errors.InvalidArgumentError(
                    f"Function {idx} contains invalid keys: {invalid_keys}")
            parameters = function['parameters']
            if not cls._check_json_schema(parameters):
                raise errors.InvalidArgumentError(
                    f"`parameters` of function {idx} is not a valid schema.")
            if parameters == {} or parameters == {'type': 'object'}:
                raise errors.InvalidArgumentError(
                    "For empty parameters, please set `type` to 'object' and `properties` to {}."
                )
            if 'responses' in function:
                responses = function['responses']
                if not cls._check_json_schema(responses):
                    raise errors.InvalidArgumentError(
                        f"`responses` of function {idx} is not a valid schema.")

    @staticmethod
    def _check_json_schema(schema: dict) -> bool:
        try:
            jsonschema.Draft202012Validator.check_schema(schema)
        except jsonschema.exceptions.SchemaError:
            return False
        else:
            return True


class ChatResponse(EBResponse):
    @property
    def is_function_response(self) -> bool:
        return hasattr(self, 'function_call')

    def get_result(self) -> Any:
        if self.is_function_response:
            return self.function_call
        else:
            return self.result

    def to_message(self) -> Dict[str, Any]:
        message: Dict[str, Any] = {'role': 'assistant'}
        if self.is_function_response:
            message['content'] = None
            message['function_call'] = self.function_call
        else:
            message['content'] = self.result
        return message
