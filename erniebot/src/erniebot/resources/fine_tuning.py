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

from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import erniebot.errors as errors
from erniebot.api_types import APIType
from erniebot.response import EBResponse
from erniebot.types import ConfigDictType, HeadersType, Request
from erniebot.utils.misc import NOT_GIVEN, NotGiven, filter_args

from .abc import Cancellable, Creatable, Queryable
from .resource import EBResource

__all__ = ["FineTuningTask", "FineTuningJob"]


class FineTuningTask(EBResource, Creatable):
    SUPPORTED_API_TYPES: ClassVar[Tuple[APIType, ...]] = (
        APIType.QIANFAN_SFT,
        APIType.AISTUDIO,
    )

    @classmethod
    def create(
        cls,
        name: str,
        description: str,
        *,
        headers: Optional[HeadersType] = None,
        request_timeout: Optional[float] = None,
        _config_: Optional[ConfigDictType] = None,
    ) -> EBResponse:
        config = _config_ or {}
        resource = cls(**config)
        kwargs = filter_args(
            name=name,
            description=description,
        )
        if headers is not None:
            kwargs["headers"] = headers
        if request_timeout is not None:
            kwargs["request_timeout"] = request_timeout
        resp = resource.create_resource(**kwargs)
        return resp

    @classmethod
    async def acreate(
        cls,
        name: str,
        description: str,
        *,
        headers: Optional[HeadersType] = None,
        request_timeout: Optional[float] = None,
        _config_: Optional[ConfigDictType] = None,
    ) -> EBResponse:
        config = _config_ or {}
        resource = cls(**config)
        kwargs = filter_args(
            name=name,
            description=description,
        )
        if headers is not None:
            kwargs["headers"] = headers
        if request_timeout is not None:
            kwargs["request_timeout"] = request_timeout
        resp = await resource.acreate_resource(**kwargs)
        return resp

    def _prepare_create(self, kwargs: Dict[str, Any]) -> Request:
        valid_keys = {"name", "description", "headers", "request_timeout"}

        invalid_keys = kwargs.keys() - valid_keys

        if len(invalid_keys) > 0:
            raise ValueError(f"Invalid keys found in `kwargs`: {list(invalid_keys)}")

        # name
        if "name" not in kwargs:
            raise errors.ArgumentNotFoundError("name")
        name = kwargs["name"]

        # description
        if "description" not in kwargs:
            raise errors.ArgumentNotFoundError("description")
        description = kwargs["description"]

        # path
        if self.api_type is APIType.QIANFAN_SFT:
            path = "/finetune/createTask"
        else:
            raise errors.UnsupportedAPITypeError(
                f"Supported API types: {self.get_supported_api_type_names()}"
            )

        # params
        params = {}
        params["name"] = name
        params["description"] = description

        # headers
        headers: HeadersType = kwargs.get("headers", {})

        # request_timeout
        request_timeout = kwargs.get("request_timeout", None)

        return Request(
            method="POST",
            path=path,
            params=params,
            headers=headers,
            timeout=request_timeout,
        )


class FineTuningJob(EBResource, Creatable, Queryable, Cancellable):
    SUPPORTED_API_TYPES: ClassVar[Tuple[APIType, ...]] = (APIType.QIANFAN_SFT,)

    @classmethod
    def create(
        cls,
        task_id: int,
        train_mode: str,
        peft_type: str,
        train_config: dict,
        train_set: List[dict],
        train_set_rate: float,
        *,
        description: Union[str, NotGiven] = NOT_GIVEN,
        headers: Optional[HeadersType] = None,
        request_timeout: Optional[float] = None,
        _config_: Optional[ConfigDictType] = None,
    ) -> EBResponse:
        config = _config_ or {}
        resource = cls(**config)
        kwargs = filter_args(
            task_id=task_id,
            train_mode=train_mode,
            peft_type=peft_type,
            train_config=train_config,
            train_set=train_set,
            train_set_rate=train_set_rate,
            description=description,
        )
        if headers is not None:
            kwargs["headers"] = headers
        if request_timeout is not None:
            kwargs["request_timeout"] = request_timeout
        resp = resource.create_resource(**kwargs)
        return resp

    @classmethod
    async def acreate(
        cls,
        task_id: int,
        train_mode: str,
        peft_type: str,
        train_config: dict,
        train_set: List[dict],
        train_set_rate: float,
        *,
        description: Union[str, NotGiven] = NOT_GIVEN,
        headers: Optional[HeadersType] = None,
        request_timeout: Optional[float] = None,
        _config_: Optional[ConfigDictType] = None,
    ) -> EBResponse:
        config = _config_ or {}
        resource = cls(**config)
        kwargs = filter_args(
            task_id=task_id,
            train_mode=train_mode,
            peft_type=peft_type,
            train_config=train_config,
            train_set=train_set,
            train_set_rate=train_set_rate,
            description=description,
        )
        if headers is not None:
            kwargs["headers"] = headers
        if request_timeout is not None:
            kwargs["request_timeout"] = request_timeout
        resp = await resource.acreate_resource(**kwargs)
        return resp

    @classmethod
    def query(
        cls,
        task_id: int,
        job_id: int,
        *,
        headers: Optional[HeadersType] = None,
        request_timeout: Optional[float] = None,
        _config_: Optional[ConfigDictType] = None,
    ) -> EBResponse:
        config = _config_ or {}
        resource = cls(**config)
        kwargs = filter_args(
            task_id=task_id,
            job_id=job_id,
        )
        if headers is not None:
            kwargs["headers"] = headers
        if request_timeout is not None:
            kwargs["request_timeout"] = request_timeout
        resp = resource.query_resource(**kwargs)
        return resp

    @classmethod
    async def aquery(
        cls,
        task_id: int,
        job_id: int,
        *,
        headers: Optional[HeadersType] = None,
        request_timeout: Optional[float] = None,
        _config_: Optional[ConfigDictType] = None,
    ) -> EBResponse:
        config = _config_ or {}
        resource = cls(**config)
        kwargs = filter_args(
            task_id=task_id,
            job_id=job_id,
        )
        if headers is not None:
            kwargs["headers"] = headers
        if request_timeout is not None:
            kwargs["request_timeout"] = request_timeout
        resp = await resource.aquery_resource(**kwargs)
        return resp

    @classmethod
    def cancel(
        cls,
        task_id: int,
        job_id: int,
        *,
        headers: Optional[HeadersType] = None,
        request_timeout: Optional[float] = None,
        _config_: Optional[ConfigDictType] = None,
    ) -> EBResponse:
        config = _config_ or {}
        resource = cls(**config)
        kwargs = filter_args(
            task_id=task_id,
            job_id=job_id,
        )
        if headers is not None:
            kwargs["headers"] = headers
        if request_timeout is not None:
            kwargs["request_timeout"] = request_timeout
        resp = resource.cancel_resource(**kwargs)
        return resp

    @classmethod
    async def acancel(
        cls,
        task_id: int,
        job_id: int,
        *,
        headers: Optional[HeadersType] = None,
        request_timeout: Optional[float] = None,
        _config_: Optional[ConfigDictType] = None,
    ) -> EBResponse:
        config = _config_ or {}
        resource = cls(**config)
        kwargs = filter_args(
            task_id=task_id,
            job_id=job_id,
        )
        if headers is not None:
            kwargs["headers"] = headers
        if request_timeout is not None:
            kwargs["request_timeout"] = request_timeout
        resp = await resource.acancel_resource(**kwargs)
        return resp

    def _prepare_create(self, kwargs: Dict[str, Any]) -> Request:
        def _get_required_arg(key: str) -> Any:
            if key not in kwargs:
                raise errors.ArgumentNotFoundError(key)
            return kwargs[key]

        valid_keys = {
            "task_id",
            "train_mode",
            "peft_type",
            "train_config",
            "train_set",
            "train_set_rate",
            "description",
            "headers",
            "request_timeout",
        }

        invalid_keys = kwargs.keys() - valid_keys

        if len(invalid_keys) > 0:
            raise ValueError(f"Invalid keys found in `kwargs`: {list(invalid_keys)}")

        # task_id
        task_id = _get_required_arg("task_id")

        # train_mode
        train_mode = _get_required_arg("train_mode")

        # peft_type
        peft_type = _get_required_arg("peft_type")

        # train_config
        train_config = _get_required_arg("train_config")

        # train_set
        train_set = _get_required_arg("train_set")

        # train_set_rate
        train_set_rate = _get_required_arg("train_set_rate")

        # path
        if self.api_type is APIType.QIANFAN_SFT:
            path = "/finetune/createJob"
        else:
            raise errors.UnsupportedAPITypeError(
                f"Supported API types: {self.get_supported_api_type_names()}"
            )

        # params
        params = {}
        params["taskId"] = task_id
        params["trainMode"] = train_mode
        params["peftType"] = peft_type
        params["trainConfig"] = train_config
        params["trainset"] = train_set
        params["trainsetRate"] = train_set_rate
        if "description" in kwargs:
            params["description"] = kwargs["description"]
        params["baseTrainType"] = "ERNIE-Bot-turbo"
        params["trainType"] = "ERNIE-Bot-turbo-0725"

        # headers
        headers: HeadersType = kwargs.get("headers", {})

        # request_timeout
        request_timeout = kwargs.get("request_timeout", None)

        return Request(
            method="POST",
            path=path,
            params=params,
            headers=headers,
            timeout=request_timeout,
        )

    def _prepare_query(self, kwargs: Dict[str, Any]) -> Request:
        valid_keys = {"task_id", "job_id", "headers", "request_timeout"}

        invalid_keys = kwargs.keys() - valid_keys

        if len(invalid_keys) > 0:
            raise ValueError(f"Invalid keys found in `kwargs`: {list(invalid_keys)}")

        # task_id
        if "task_id" not in kwargs:
            raise errors.ArgumentNotFoundError("task_id")
        task_id = kwargs["task_id"]

        # job_id
        if "job_id" not in kwargs:
            raise errors.ArgumentNotFoundError("job_id")
        job_id = kwargs["job_id"]

        # path
        if self.api_type is APIType.QIANFAN_SFT:
            path = "/finetune/jobDetail"
        else:
            raise errors.UnsupportedAPITypeError(
                f"Supported API types: {self.get_supported_api_type_names()}"
            )

        # params
        params = {}
        params["taskId"] = task_id
        params["jobId"] = job_id

        # headers
        headers: HeadersType = kwargs.get("headers", {})

        # request_timeout
        request_timeout = kwargs.get("request_timeout", None)

        return Request(
            method="POST",
            path=path,
            params=params,
            headers=headers,
            timeout=request_timeout,
        )

    def _prepare_cancel(self, kwargs: Dict[str, Any]) -> Request:
        valid_keys = {"task_id", "job_id", "headers", "request_timeout"}

        invalid_keys = kwargs.keys() - valid_keys

        if len(invalid_keys) > 0:
            raise ValueError(f"Invalid keys found in `kwargs`: {list(invalid_keys)}")

        # task_id
        if "task_id" not in kwargs:
            raise errors.ArgumentNotFoundError("task_id")
        task_id = kwargs["task_id"]

        # job_id
        if "job_id" not in kwargs:
            raise errors.ArgumentNotFoundError("job_id")
        job_id = kwargs["job_id"]

        # path
        if self.api_type is APIType.QIANFAN_SFT:
            path = "/finetune/stopJob"
        else:
            raise errors.UnsupportedAPITypeError(
                f"Supported API types: {self.get_supported_api_type_names()}"
            )

        # params
        params = {}
        params["taskId"] = task_id
        params["jobId"] = job_id

        # headers
        headers: HeadersType = kwargs.get("headers", {})

        # request_timeout
        request_timeout = kwargs.get("request_timeout", None)

        return Request(
            method="POST",
            path=path,
            params=params,
            headers=headers,
            timeout=request_timeout,
        )
