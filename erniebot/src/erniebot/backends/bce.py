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

import asyncio
import datetime
import hashlib
import hmac
import urllib.parse
from typing import (
    AsyncIterator,
    ClassVar,
    Dict,
    Final,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import erniebot.errors as errors
import erniebot.utils.logging as logging
from erniebot.api_types import APIType
from erniebot.auth import build_auth_token_manager
from erniebot.response import EBResponse
from erniebot.types import ConfigDictType, HeadersType, ParamsType
from erniebot.utils.url import add_query_params

from .base import EBBackend


class _BCELegacyBackend(EBBackend):
    def __init__(self, config_dict: ConfigDictType) -> None:
        super().__init__(config_dict=config_dict)
        self._auth_manager = build_auth_token_manager(
            "bce",
            self.api_type,
            auth_token=self._cfg["access_token"],
            ak=self._cfg["ak"],
            sk=self._cfg["sk"],
        )

    def request(
        self,
        method: str,
        path: str,
        stream: bool,
        *,
        params: Optional[ParamsType] = None,
        headers: Optional[HeadersType] = None,
        request_timeout: Optional[float] = None,
    ) -> Union[EBResponse, Iterator[EBResponse]]:
        url, headers, data = self._client.prepare_request(
            method,
            path,
            supplied_headers=headers,
            params=params,
        )

        access_token = self._auth_manager.get_auth_token()
        url_with_token = add_query_params(url, [("access_token", access_token)])
        try:
            return self._client.send_request(
                method,
                url_with_token,
                stream,
                data=data,
                headers=headers,
                request_timeout=request_timeout,
            )
        except (errors.TokenExpiredError, errors.InvalidTokenError):
            logging.warning(
                "The access token provided is invalid or has expired."
                " An automatic update will be performed before retrying."
            )
            access_token = self._auth_manager.update_auth_token()
            url_with_token = add_query_params(url, [("access_token", access_token)])
            return self._client.send_request(
                method,
                url_with_token,
                stream,
                data=data,
                headers=headers,
                request_timeout=request_timeout,
            )

    async def arequest(
        self,
        method: str,
        path: str,
        stream: bool,
        *,
        params: Optional[ParamsType] = None,
        headers: Optional[HeadersType] = None,
        request_timeout: Optional[float] = None,
    ) -> Union[EBResponse, AsyncIterator[EBResponse]]:
        url, headers, data = self._client.prepare_request(
            method,
            path,
            supplied_headers=headers,
            params=params,
        )

        loop = asyncio.get_running_loop()
        # XXX: The default executor is used.
        access_token = await loop.run_in_executor(None, self._auth_manager.get_auth_token)
        url_with_token = add_query_params(url, [("access_token", access_token)])
        try:
            return await self._client.asend_request(
                method,
                url_with_token,
                stream,
                data=data,
                headers=headers,
                request_timeout=request_timeout,
            )
        except (errors.TokenExpiredError, errors.InvalidTokenError):
            logging.warning(
                "The access token provided is invalid or has expired."
                " An automatic update will be performed before retrying."
            )
            # XXX: The default executor is used.
            access_token = await loop.run_in_executor(None, self._auth_manager.update_auth_token)
            url_with_token = add_query_params(url, [("access_token", access_token)])
            return await self._client.asend_request(
                method,
                url_with_token,
                stream,
                data=data,
                headers=headers,
                request_timeout=request_timeout,
            )


class _BCEBackend(EBBackend):
    _SIG_EXPIRATION_IN_SECS: Final[float] = 1800

    def __init__(self, config_dict: ConfigDictType) -> None:
        super().__init__(config_dict=config_dict)
        ak = self._cfg.get("ak")
        sk = self._cfg.get("sk")
        if ak is None or sk is None:
            raise RuntimeError("Invalid access key ID or secret access key")
        self._ak = ak
        self._sk = sk

    def request(
        self,
        method: str,
        path: str,
        stream: bool,
        *,
        params: Optional[ParamsType] = None,
        headers: Optional[HeadersType] = None,
        request_timeout: Optional[float] = None,
    ) -> Union[EBResponse, Iterator[EBResponse]]:
        url, headers, data = self._client.prepare_request(
            method,
            path,
            supplied_headers=headers,
            params=params,
        )
        headers = self._add_bce_fields_to_headers(headers, method, url)
        return self._client.send_request(
            method,
            url,
            stream,
            data=data,
            headers=headers,
            request_timeout=request_timeout,
        )

    async def arequest(
        self,
        method: str,
        path: str,
        stream: bool,
        *,
        params: Optional[ParamsType] = None,
        headers: Optional[HeadersType] = None,
        request_timeout: Optional[float] = None,
    ) -> Union[EBResponse, AsyncIterator[EBResponse]]:
        url, headers, data = self._client.prepare_request(
            method,
            path,
            supplied_headers=headers,
            params=params,
        )
        headers = self._add_bce_fields_to_headers(headers, method, url)
        return await self._client.asend_request(
            method,
            url,
            stream,
            data=data,
            headers=headers,
            request_timeout=request_timeout,
        )

    def _add_bce_fields_to_headers(self, headers: HeadersType, method: str, url: str) -> HeadersType:
        host, path, query_params = self._get_url_parts(url)
        headers["Host"] = urllib.parse.quote(host)
        x_bce_date = self._get_canonical_time()
        headers["x-bce-date"] = x_bce_date
        credentials = {"ak": self._ak, "sk": self._sk}
        headers["Authorization"] = self._sign(
            credentials=credentials,
            method=method,
            path=path,
            headers=headers,
            params=query_params,
            timestamp=x_bce_date,
            headers_to_sign=None,
        )
        return headers

    def _sign(
        self,
        credentials: Dict[str, str],
        method: str,
        path: str,
        headers: HeadersType,
        params: Dict[str, List[str]],
        timestamp: str,
        headers_to_sign: Optional[List[str]] = None,
    ) -> str:
        auth_str_prefix = (
            "bce-auth-v1"
            + "/"
            + credentials["ak"]
            + "/"
            + timestamp
            + "/"
            + str(self._SIG_EXPIRATION_IN_SECS)
        )

        method = method.upper()
        canonical_uri = urllib.parse.quote(path)
        if headers_to_sign is None:
            headers_to_sign = ["content-type", "host", "x-bce-date"]
        canonical_header_list = []
        for key, val in headers.items():
            key = key.lower()
            if key in headers_to_sign:
                val = val.strip()
                if len(val) > 0:
                    key = urllib.parse.quote(key, safe="")
                    val = urllib.parse.quote(val, safe="")
                    header = key + ":" + val
                    canonical_header_list.append(header)
        canonical_header_list.sort()
        canonical_headers = "\n".join(canonical_header_list)
        signed_headers = ";".join(headers_to_sign)
        canonical_query_list = []
        for key, val_list in params.items():
            if len(val_list) > 1:
                raise ValueError(f"Name {repr(key)} has multiple values.")
            key = urllib.parse.quote(key, safe="")
            val = urllib.parse.quote(val_list[0], safe="")
            canonical_query_list.append(key + "=" + val)
        canonical_query_list.sort()
        canonical_query_str = "&".join(canonical_query_list)
        canonical_request = (
            method + "\n" + canonical_uri + "\n" + canonical_query_str + "\n" + canonical_headers
        )

        signing_key = hmac.new(
            credentials["sk"].encode("utf-8"),
            auth_str_prefix.encode("utf-8"),
            hashlib.sha256,
        )
        signature = hmac.new(
            signing_key.hexdigest().encode("utf-8"),
            canonical_request.encode("utf-8"),
            hashlib.sha256,
        )

        return auth_str_prefix + "/" + signed_headers + "/" + signature.hexdigest()

    def _get_canonical_time(self, timestamp: int = 0) -> str:
        if timestamp == 0:
            utctime = datetime.datetime.utcnow()
        else:
            utctime = datetime.datetime.utcfromtimestamp(timestamp)
        return "%04d-%02d-%02dT%02d:%02d:%02dZ" % (
            utctime.year,
            utctime.month,
            utctime.day,
            utctime.hour,
            utctime.minute,
            utctime.second,
        )

    def _get_url_parts(self, url: str) -> Tuple[str, str, Dict[str, List[str]]]:
        res = urllib.parse.urlparse(url)
        host = res.netloc
        path = res.path
        query = res.query
        if len(query) > 0:
            params = urllib.parse.parse_qs(query, keep_blank_values=True, strict_parsing=True)
        else:
            params = {}
        return host, path, params


class QianfanLegacyBackend(_BCELegacyBackend):
    api_type: ClassVar[APIType] = APIType.QIANFAN
    base_url: ClassVar[str] = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop"

    @classmethod
    def handle_response(cls, resp: EBResponse) -> EBResponse:
        if "error_code" in resp and "error_msg" in resp:
            ecode = resp["error_code"]
            emsg = resp["error_msg"]
            if ecode == 17:
                raise errors.RequestLimitError(emsg, ecode=ecode)
            elif ecode == 18:
                raise errors.RateLimitError(emsg, ecode=ecode)
            elif ecode == 110:
                raise errors.InvalidTokenError(emsg, ecode=ecode)
            elif ecode == 111:
                raise errors.TokenExpiredError(emsg, ecode=ecode)
            elif ecode in (336002, 336003, 336006, 336007, 336102):
                raise errors.BadRequestError(emsg, ecode=ecode)
            elif ecode == 336100:
                raise errors.TryAgain(emsg, ecode=ecode)
            else:
                raise errors.APIError(emsg, ecode=ecode)
        else:
            return resp


class YinianBackend(_BCELegacyBackend):
    api_type: ClassVar[APIType] = APIType.YINIAN
    base_url: ClassVar[str] = "https://aip.baidubce.com/rpc/2.0/ernievilg/v1"

    @classmethod
    def handle_response(cls, resp: EBResponse) -> EBResponse:
        if "error_code" in resp and "error_msg" in resp:
            ecode = resp["error_code"]
            emsg = resp["error_msg"]
            print(ecode)
            if ecode in (4, 17):
                raise errors.RequestLimitError(emsg, ecode=ecode)
            elif ecode in (13, 15, 18):
                raise errors.RateLimitError(emsg, ecode=ecode)
            elif ecode == 110:
                raise errors.InvalidTokenError(emsg, ecode=ecode)
            elif ecode == 111:
                raise errors.TokenExpiredError(emsg, ecode=ecode)
            elif ecode in (216100, 282003, 282004):
                raise errors.BadRequestError(emsg, ecode=ecode)
            else:
                raise errors.APIError(emsg, ecode=ecode)
        else:
            return resp


class QianfanBackend(_BCEBackend):
    api_type: ClassVar[APIType] = APIType.QIANFAN
    base_url: ClassVar[str] = "https://qianfan.baidubce.com/wenxinworkshop"

    @classmethod
    def handle_response(cls, resp: EBResponse) -> EBResponse:
        if "error_code" in resp and "error_msg" in resp:
            ecode = resp["error_code"]
            emsg = resp["error_msg"]
            if ecode == 500001:
                raise errors.BadRequestError(emsg, ecode=ecode)
            else:
                raise errors.APIError(emsg, ecode=ecode)
        else:
            return resp
