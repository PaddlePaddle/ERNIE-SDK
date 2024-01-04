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

import functools
import http
import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Final, Hashable, Optional, Tuple

import requests

from . import errors
from .api_types import APIType
from .utils import logging
from .utils.misc import SingletonMeta

__all__ = ["build_auth_token_manager"]


def build_auth_token_manager(manager_type: str, api_type: APIType, **kwargs: Any) -> "AuthTokenManager":
    if manager_type == "bce":
        return BCEAuthTokenManager(api_type, **kwargs)
    else:
        raise ValueError(f"Unsupported manager type: {manager_type}")


class _GlobalAuthTokenCache(metaclass=SingletonMeta):
    _MIN_UPDATE_INTERVAL_SECS: Final[float] = 3600

    @dataclass
    class _Record(object):
        updated_at: Optional[float]
        auth_token: Optional[str]
        lock: threading.Lock

    def __init__(self) -> None:
        super().__init__()
        self._cache: Dict[Tuple[str, Hashable], _GlobalAuthTokenCache._Record] = dict()
        self._lock = threading.Lock()

    def retrieve_auth_token(self, api_type: str, key: Hashable) -> Optional[str]:
        key_pair = self._constr_key_pair(api_type, key)

        with self._lock:
            record = self._cache.get(key_pair, None)

        if record is not None:
            with record.lock:
                auth_token = record.auth_token
        else:
            auth_token = None

        return auth_token

    def upsert_auth_token(
        self, api_type: str, key: Hashable, token_requestor: Callable[[], str]
    ) -> Tuple[str, bool]:
        key_pair = self._constr_key_pair(api_type, key)

        with self._lock:
            record = self._cache.get(key_pair, None)
            if record is None:
                self._cache[key_pair] = _GlobalAuthTokenCache._Record(
                    auth_token=None, updated_at=None, lock=threading.Lock()
                )
                record = self._cache[key_pair]

        with record.lock:
            timestamp = time.monotonic()
            if record.updated_at is None or timestamp - record.updated_at > self._MIN_UPDATE_INTERVAL_SECS:
                try:
                    auth_token = token_requestor()
                except Exception as e:
                    raise errors.TokenUpdateFailedError from e
                record.auth_token = auth_token
                record.updated_at = time.monotonic()
                upserted = True
            else:
                assert record.auth_token is not None
                auth_token = record.auth_token
                upserted = False

        return auth_token, upserted

    def _constr_key_pair(self, key1: str, key2: Hashable) -> Tuple[str, Hashable]:
        return (key1, key2)


class AuthTokenManager(object):
    def __init__(self, api_type: APIType, *, auth_token: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__()
        self.api_type = api_type
        self._cfg = dict(**kwargs)
        self._cache = _GlobalAuthTokenCache()
        self._cache_key = self._get_cache_key()
        self._token = auth_token

    def get_auth_token(self) -> str:
        if self._token is None:
            self._token = self._init_auth_token()
        return self._token

    def update_auth_token(self) -> str:
        new_token = self._update_cache(init=False)
        self._token = new_token
        logging.info("Security token has been updated.")
        return self._token

    def _request_auth_token(self, init: bool) -> str:
        raise NotImplementedError

    def _get_cache_key(self) -> Hashable:
        raise NotImplementedError

    def _init_auth_token(self) -> str:
        cached_token = self._retrieve_from_cache()
        if cached_token is not None:
            logging.info("Cached security token will be used.")
            token = cached_token
        else:
            logging.info(
                "Security token is not set. It will be retrieved or generated based on other parameters."
            )
            token = self._update_cache(init=True)
        return token

    def _retrieve_from_cache(self) -> Optional[str]:
        return self._cache.retrieve_auth_token(self.api_type.name, self._cache_key)

    def _update_cache(self, init: bool) -> str:
        token, upserted = self._cache.upsert_auth_token(
            self.api_type.name,
            self._cache_key,
            functools.partial(self._request_auth_token, init=init),
        )
        if upserted:
            logging.debug("Cache updated")
        return token


class BCEAuthTokenManager(AuthTokenManager):
    def __init__(
        self,
        api_type: APIType,
        *,
        auth_token: Optional[str] = None,
        ak: Optional[str] = None,
        sk: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(api_type, auth_token=auth_token, ak=ak, sk=sk, **kwargs)

    def _request_auth_token(self, init: bool) -> str:
        # `init` not used
        url = "https://aip.baidubce.com/oauth/2.0/token"
        ak = self._cfg["ak"]
        sk = self._cfg["sk"]
        if ak is None or sk is None:
            raise RuntimeError("Invalid API key or secret key")
        params = {
            "grant_type": "client_credentials",
            "client_id": ak,
            "client_secret": sk,
        }
        result = requests.request(method="GET", url=url, params=params, timeout=3)
        if result.status_code != http.HTTPStatus.OK:
            raise errors.HTTPRequestError(
                f"Status code is not {http.HTTPStatus.OK}.",
                rcode=result.status_code,
                rbody=result.content.decode("utf-8"),
                rheaders=result.headers,
            )
        else:
            rbody = result.content.decode("utf-8")
            rbody = json.loads(rbody)
            if not isinstance(rbody, dict):
                raise errors.HTTPRequestError("The response body cannot be deserialized to a dict.")
            token = rbody["access_token"]
            return token

    def _get_cache_key(self) -> Hashable:
        return (self._cfg["ak"], self._cfg["sk"])
