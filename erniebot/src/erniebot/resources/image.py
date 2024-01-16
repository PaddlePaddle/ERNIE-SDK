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

from typing import Any, ClassVar, Dict, Optional, Tuple, Union

from typing_extensions import TypeAlias

import erniebot.errors as errors
from erniebot.api_types import APIType
from erniebot.response import EBResponse
from erniebot.types import ConfigDictType, HeadersType, Request
from erniebot.utils.misc import NOT_GIVEN, NotGiven, filter_args

from .resource import EBResource

__all__ = ["Image", "ImageResponse", "ImageV1", "ImageV2", "ImageV2Response"]


class _Image(EBResource):
    def create_resource(self, **create_kwargs: Any) -> EBResponse:
        req = self._prepare_paint(create_kwargs)
        timeout = req.timeout
        resp_p = self.request(
            method=req.method,
            path=req.path,
            stream=False,
            params=req.params,
            headers=req.headers,
            request_timeout=timeout,
        )

        req = self._prepare_fetch(resp_p)
        resp_f = self.poll(
            until=self._check_status,
            method=req.method,
            path=req.path,
            params=req.params,
            headers=req.headers,
            # XXX: Reuse `timeout`. Should we implement finer-grained control?
            request_timeout=timeout,
        )

        resp_f = self._postprocess(resp_f)

        return resp_f

    async def acreate_resource(self, **create_kwargs: Any) -> EBResponse:
        req = self._prepare_paint(create_kwargs)
        timeout = req.timeout
        resp_p = await self.arequest(
            method=req.method,
            path=req.path,
            stream=False,
            params=req.params,
            headers=req.headers,
            request_timeout=timeout,
        )

        req = self._prepare_fetch(resp_p)
        resp_f = await self.apoll(
            until=self._check_status,
            method=req.method,
            path=req.path,
            params=req.params,
            headers=req.headers,
            request_timeout=timeout,
        )

        resp_f = self._postprocess(resp_f)

        return resp_f

    def _prepare_paint(self, kwargs: Dict[str, Any]) -> Request:
        raise NotImplementedError

    def _prepare_fetch(self, resp_p: EBResponse) -> Request:
        raise NotImplementedError

    def _postprocess(self, resp_f: EBResponse) -> EBResponse:
        raise NotImplementedError

    @staticmethod
    def _check_status(resp: EBResponse) -> bool:
        raise NotImplementedError


class ImageV1(_Image):
    SUPPORTED_API_TYPES: ClassVar[Tuple[APIType, ...]] = (APIType.YINIAN,)

    @classmethod
    def create(
        cls,
        text: str,
        resolution: str,
        style: str,
        *,
        num: Union[int, NotGiven] = NOT_GIVEN,
        headers: Optional[HeadersType] = None,
        request_timeout: Optional[float] = None,
        _config_: Optional[ConfigDictType] = None,
    ) -> EBResponse:
        config = _config_ or {}
        resource = cls(**config)
        kwargs = filter_args(
            text=text,
            resolution=resolution,
            style=style,
            num=num,
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
        text: str,
        resolution: str,
        style: str,
        *,
        num: Union[int, NotGiven] = NOT_GIVEN,
        headers: Optional[HeadersType] = None,
        request_timeout: Optional[float] = None,
        _config_: Optional[ConfigDictType] = None,
    ) -> EBResponse:
        config = _config_ or {}
        resource = cls(**config)
        kwargs = filter_args(
            text=text,
            resolution=resolution,
            style=style,
            num=num,
        )
        if headers is not None:
            kwargs["headers"] = headers
        if request_timeout is not None:
            kwargs["request_timeout"] = request_timeout
        resp = await resource.acreate_resource(**kwargs)
        return resp

    def _prepare_paint(self, kwargs: Dict[str, Any]) -> Request:
        def _set_val_if_key_exists(src: dict, dst: dict, key: str) -> None:
            if key in src:
                dst[key] = src[key]

        valid_keys = {
            "text",
            "resolution",
            "style",
            "num",
            "headers",
            "request_timeout",
        }

        invalid_keys = kwargs.keys() - valid_keys

        if len(invalid_keys) > 0:
            raise ValueError(f"Invalid keys found in `kwargs`: {list(invalid_keys)}")

        # text
        if "text" not in kwargs:
            raise errors.ArgumentNotFoundError("text")
        text = kwargs["text"]

        # resolution
        if "resolution" not in kwargs:
            raise errors.ArgumentNotFoundError("resolution")
        resolution = kwargs["resolution"]

        # style
        if "style" not in kwargs:
            raise errors.ArgumentNotFoundError("style")
        style = kwargs["style"]

        # path
        if self.api_type is APIType.YINIAN:
            path = "/txt2img"
        else:
            raise errors.UnsupportedAPITypeError(
                f"Supported API types: {self.get_supported_api_type_names()}"
            )

        # params
        params = {}
        params["text"] = text
        params["resolution"] = resolution
        params["style"] = style
        _set_val_if_key_exists(kwargs, params, "num")

        # headers
        headers: HeadersType = {}
        if self.api_type is APIType.YINIAN:
            headers["Accept"] = "application/json"
        if "headers" in kwargs:
            headers.update(kwargs["headers"])

        # request_timeout
        request_timeout = kwargs.get("request_timeout", None)

        return Request(
            method="POST",
            path=path,
            params=params,
            headers=headers,
            timeout=request_timeout,
        )

    def _prepare_fetch(self, resp_p: EBResponse) -> Request:
        # path
        if self.api_type is APIType.YINIAN:
            path = "/getImg"
        else:
            raise errors.UnsupportedAPITypeError(
                f"Supported API types: {self.get_supported_api_type_names()}"
            )

        # params
        params = {}
        params["taskId"] = resp_p.data["taskId"]

        # headers
        headers: HeadersType = {}
        if self.api_type is APIType.YINIAN:
            headers["Accept"] = "application/json"

        return Request(
            method="POST",
            path=path,
            params=params,
            headers=headers,
        )

    def _postprocess(self, resp_f: EBResponse) -> EBResponse:
        return resp_f

    @staticmethod
    def _check_status(resp: EBResponse) -> bool:
        status = resp.data["status"]
        return status == 1


class ImageV2(_Image):
    SUPPORTED_API_TYPES: ClassVar[Tuple[APIType, ...]] = (APIType.YINIAN,)

    @classmethod
    def create(
        cls,
        model: str,
        prompt: str,
        width: int,
        height: int,
        *,
        version: Union[str, NotGiven] = NOT_GIVEN,
        image_num: Union[int, NotGiven] = NOT_GIVEN,
        headers: Optional[HeadersType] = None,
        request_timeout: Optional[float] = None,
        _config_: Optional[ConfigDictType] = None,
    ) -> "ImageV2Response":
        """Creates images based on the given prompt.

        Args:
            model: Name of the model to use.
            prompt: Text that describes the image(s).
            width: Width of the image(s).
            height: Height of the image(s).
            version: Version of the model.
            image_num: Number of images to generate.
            headers: Custom headers to send with the request.
            request_timeout: Timeout for a single request.
            _config_: Overrides the global settings.

        Returns:
            Response containing the image URLs.
        """
        config = _config_ or {}
        resource = cls(**config)
        kwargs = filter_args(
            model=model,
            prompt=prompt,
            width=width,
            height=height,
            version=version,
            image_num=image_num,
        )
        if headers is not None:
            kwargs["headers"] = headers
        if request_timeout is not None:
            kwargs["request_timeout"] = request_timeout
        resp = resource.create_resource(**kwargs)
        return ImageV2Response.from_mapping(resp)

    @classmethod
    async def acreate(
        cls,
        model: str,
        prompt: str,
        width: int,
        height: int,
        *,
        version: Union[str, NotGiven] = NOT_GIVEN,
        image_num: Union[int, NotGiven] = NOT_GIVEN,
        headers: Optional[HeadersType] = None,
        request_timeout: Optional[float] = None,
        _config_: Optional[ConfigDictType] = None,
    ) -> "ImageV2Response":
        """Creates images based on the given prompt.

        Args:
            model: Name of the model to use.
            prompt: Text that describes the image(s).
            width: Width of the image(s).
            height: Height of the image(s).
            version: Version of the model.
            image_num: Number of images to generate.
            headers: Custom headers to send with the request.
            request_timeout: Timeout for a single request.
            _config_: Overrides the global settings.

        Returns:
            Response containing the URLs of the generated images.
        """
        config = _config_ or {}
        resource = cls(**config)
        kwargs = filter_args(
            model=model,
            prompt=prompt,
            width=width,
            height=height,
            version=version,
            image_num=image_num,
        )
        if headers is not None:
            kwargs["headers"] = headers
        if request_timeout is not None:
            kwargs["request_timeout"] = request_timeout
        resp = await resource.acreate_resource(**kwargs)
        return ImageV2Response.from_mapping(resp)

    def _prepare_paint(self, kwargs: Dict[str, Any]) -> Request:
        def _set_val_if_key_exists(src: dict, dst: dict, key: str) -> None:
            if key in src:
                dst[key] = src[key]

        valid_keys = {
            "model",
            "prompt",
            "width",
            "height",
            "version",
            "image_num",
            "headers",
            "request_timeout",
        }

        invalid_keys = kwargs.keys() - valid_keys

        if len(invalid_keys) > 0:
            raise ValueError(f"Invalid keys found in `kwargs`: {list(invalid_keys)}")

        # model
        if "model" not in kwargs:
            raise errors.ArgumentNotFoundError("model")
        model = kwargs["model"]

        # prompt
        if "prompt" not in kwargs:
            raise errors.ArgumentNotFoundError("prompt")
        prompt = kwargs["prompt"]

        # width
        if "width" not in kwargs:
            raise errors.ArgumentNotFoundError("width")
        width = kwargs["width"]

        # height
        if "height" not in kwargs:
            raise errors.ArgumentNotFoundError("height")
        height = kwargs["height"]

        # path
        if self.api_type is APIType.YINIAN:
            path = "/txt2imgv2"
            if model != "ernie-vilg-v2":
                raise errors.InvalidArgumentError(f"{repr(model)} is not a supported model.")
        else:
            raise errors.UnsupportedAPITypeError(
                f"Supported API types: {self.get_supported_api_type_names()}"
            )

        # params
        params = {}
        params["prompt"] = prompt
        params["width"] = width
        params["height"] = height
        _set_val_if_key_exists(kwargs, params, "version")
        _set_val_if_key_exists(kwargs, params, "image_num")

        # headers
        headers: HeadersType = {}
        if self.api_type is APIType.YINIAN:
            headers["Accept"] = "application/json"
        if "headers" in kwargs:
            headers.update(kwargs["headers"])

        # request_timeout
        request_timeout = kwargs.get("request_timeout", None)

        return Request(
            method="POST",
            path=path,
            params=params,
            headers=headers,
            timeout=request_timeout,
        )

    def _prepare_fetch(self, resp_p: EBResponse) -> Request:
        # path
        if self.api_type is APIType.YINIAN:
            path = "/getImgv2"
        else:
            raise errors.UnsupportedAPITypeError(
                f"Supported API types: {self.get_supported_api_type_names()}"
            )

        # params
        params = {}
        params["task_id"] = resp_p.data["task_id"]

        # headers
        headers: HeadersType = {}
        if self.api_type is APIType.YINIAN:
            headers["Accept"] = "application/json"

        return Request(
            method="POST",
            path=path,
            params=params,
            headers=headers,
        )

    def _postprocess(self, resp_f: EBResponse) -> EBResponse:
        return ImageV2Response.from_mapping(resp_f)

    @staticmethod
    def _check_status(resp: EBResponse) -> bool:
        status = resp.data["task_status"]
        if status == "FAILED":
            raise errors.APIError("Image generation failed.")
        return status == "SUCCESS"


class ImageV2Response(EBResponse):
    def get_result(self) -> Any:
        image_urls = []
        for task_item in self.data["sub_task_result_list"]:
            for image_item in task_item["final_image_list"]:
                review_conclusion = image_item["img_approve_conclusion"]
                if review_conclusion == "pass":
                    image_urls.append(image_item["img_url"])
        return image_urls


Image: TypeAlias = ImageV2
ImageResponse: TypeAlias = ImageV2Response
