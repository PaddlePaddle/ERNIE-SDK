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
from erniebot.response import EBResponse
from erniebot.types import (FilesType, HeadersType, ParamsType, ResponseT)
from erniebot.utils.misc import transform
from .abc import Cancellable, Creatable, Queryable
from .resource import EBResource


class FineTuningTask(EBResource, Creatable):
    SUPPORTED_API_TYPES: ClassVar[Tuple[APIType, ...]] = (APIType.QIANFAN_SFT,
                                                          APIType.AISTUDIO)

    def _prepare_create(self,
                        kwargs: Dict[str, Any]) -> Tuple[str,
                                                         Optional[ParamsType],
                                                         Optional[HeadersType],
                                                         Optional[FilesType],
                                                         bool,
                                                         Optional[float],
                                                         ]:
        VALID_KEYS = {'name', 'description', 'headers', 'request_timeout'}

        invalid_keys = kwargs.keys() - VALID_KEYS

        if len(invalid_keys) > 0:
            raise errors.InvalidArgumentError(
                f"Invalid keys found in `kwargs`: {list(invalid_keys)}")

        # name
        if 'name' not in kwargs:
            raise errors.ArgumentNotFoundError("`name` is not found.")
        name = kwargs['name']

        # description
        if 'description' not in kwargs:
            raise errors.ArgumentNotFoundError("`description` is not found.")
        description = kwargs['description']

        # url
        if self.api_type is APIType.QIANFAN_SFT:
            url = "/finetune/createTask"
        else:
            raise errors.UnsupportedAPITypeError(
                f"Supported API types: {self.get_supported_api_type_names()}")

        # params
        params = {}
        params['name'] = name
        params['description'] = description

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
        return transform(FineTuningResponse.from_mapping, resp)


class FineTuningJob(EBResource, Creatable, Queryable, Cancellable):
    SUPPORTED_API_TYPES: ClassVar[Tuple[APIType, ...]] = (APIType.QIANFAN_SFT, )

    def _prepare_create(self,
                        kwargs: Dict[str, Any]) -> Tuple[str,
                                                         Optional[ParamsType],
                                                         Optional[HeadersType],
                                                         Optional[FilesType],
                                                         bool,
                                                         Optional[float],
                                                         ]:
        def _get_required_arg(key: str) -> Any:
            if key not in kwargs:
                raise errors.ArgumentNotFoundError(f"`{key}` is not found.")
            return kwargs[key]

        VALID_KEYS = {
            'task_id',
            'train_mode',
            'peft_type',
            'train_config',
            'train_set',
            'train_set_rate',
            'description',
            'headers',
            'request_timeout',
        }

        invalid_keys = kwargs.keys() - VALID_KEYS

        if len(invalid_keys) > 0:
            raise errors.InvalidArgumentError(
                f"Invalid keys found in `kwargs`: {list(invalid_keys)}")

        # task_id
        task_id = _get_required_arg('task_id')

        # train_mode
        train_mode = _get_required_arg('train_mode')

        # peft_type
        peft_type = _get_required_arg('peft_type')

        # train_config
        train_config = _get_required_arg('train_config')

        # train_set
        train_set = _get_required_arg('train_set')

        # train_set_rate
        train_set_rate = _get_required_arg('train_set_rate')

        # url
        if self.api_type is APIType.QIANFAN_SFT:
            url = "/finetune/createJob"
        else:
            raise errors.UnsupportedAPITypeError(
                f"Supported API types: {self.get_supported_api_type_names()}")

        # params
        params = {}
        params['taskId'] = task_id
        params['trainMode'] = train_mode
        params['peftType'] = peft_type
        params['trainConfig'] = train_config
        params['trainset'] = train_set
        params['trainsetRate'] = train_set_rate
        if 'description' in kwargs:
            params['description'] = kwargs['description']
        params['baseTrainType'] = 'ERNIE-Bot-turbo'
        params['trainType'] = 'ERNIE-Bot-turbo-0725'

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
        return transform(FineTuningResponse.from_mapping, resp)

    def _prepare_query(self,
                       kwargs: Dict[str, Any]) -> Tuple[str,
                                                        Optional[ParamsType],
                                                        Optional[HeadersType],
                                                        Optional[float],
                                                        ]:
        VALID_KEYS = {'task_id', 'job_id', 'headers', 'request_timeout'}

        invalid_keys = kwargs.keys() - VALID_KEYS

        if len(invalid_keys) > 0:
            raise errors.InvalidArgumentError(
                f"Invalid keys found in `kwargs`: {list(invalid_keys)}")

        # task_id
        if 'task_id' not in kwargs:
            raise errors.ArgumentNotFoundError("`task_id` is not found.")
        task_id = kwargs['task_id']

        # job_id
        if 'job_id' not in kwargs:
            raise errors.ArgumentNotFoundError("`job_id` is not found.")
        job_id = kwargs['job_id']

        # url
        if self.api_type is APIType.QIANFAN_SFT:
            url = "/finetune/jobDetail"
        else:
            raise errors.UnsupportedAPITypeError(
                f"Supported API types: {self.get_supported_api_type_names()}")

        # params
        params = {}
        params['taskId'] = task_id
        params['jobId'] = job_id

        # headers
        headers = kwargs.get('headers', None)

        # request_timeout
        request_timeout = kwargs.get('request_timeout', None)

        return url, params, headers, request_timeout

    def _postprocess_query(self, resp: EBResponse) -> EBResponse:
        return FineTuningResponse.from_mapping(resp)

    def _prepare_cancel(self,
                        kwargs: Dict[str, Any]) -> Tuple[str,
                                                         Optional[ParamsType],
                                                         Optional[HeadersType],
                                                         Optional[float],
                                                         ]:
        VALID_KEYS = {'task_id', 'job_id', 'headers', 'request_timeout'}

        invalid_keys = kwargs.keys() - VALID_KEYS

        if len(invalid_keys) > 0:
            raise errors.InvalidArgumentError(
                f"Invalid keys found in `kwargs`: {list(invalid_keys)}")

        # task_id
        if 'task_id' not in kwargs:
            raise errors.ArgumentNotFoundError("`task_id` is not found.")
        task_id = kwargs['task_id']

        # job_id
        if 'job_id' not in kwargs:
            raise errors.ArgumentNotFoundError("`job_id` is not found.")
        job_id = kwargs['job_id']

        # url
        if self.api_type is APIType.QIANFAN_SFT:
            url = "/finetune/stopJob"
        else:
            raise errors.UnsupportedAPITypeError(
                f"Supported API types: {self.get_supported_api_type_names()}")

        # params
        params = {}
        params['taskId'] = task_id
        params['jobId'] = job_id

        # headers
        headers = kwargs.get('headers', None)

        # request_timeout
        request_timeout = kwargs.get('request_timeout', None)

        return url, params, headers, request_timeout

    def _postprocess_cancel(self, resp: EBResponse) -> EBResponse:
        return FineTuningResponse.from_mapping(resp)


class FineTuningResponse(EBResponse):
    def get_result(self) -> Any:
        return self.result
