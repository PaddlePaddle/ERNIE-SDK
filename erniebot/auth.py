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
from typing import (Any, Callable, Dict, Hashable, Optional, Tuple, Union)

import requests

from . import errors
from .api_types import APIType
from .utils.logging import logger
from .utils.misc import Singleton

__all__ = ['build_auth_manager']


def build_auth_manager(manager_type: str, api_type: APIType,
                       **kwargs: Any) -> 'AuthManager':
    if manager_type == 'bce':
        return BCEAuthManager(api_type, **kwargs)
    else:
        raise ValueError(f"Unsupported manager type: {manager_type}")


class _GlobalAuthCache(metaclass=Singleton):
    _MIN_UPDATE_INTERVAL: int = 60

    def __init__(self) -> None:
        super().__init__()
        self._cache: Dict[Tuple[str, Hashable], str] = dict()
        self._last_update_time: Dict[Tuple[str, Hashable], float] = dict()

        # Following condition variable, flag, and counters are used to implement a writer-preferring RW lock.
        # We use an object-level lock for simplicity.
        # For higher concurrency performance, per-key locks can be used instead.
        self._cond = threading.Condition(threading.Lock())
        self._is_writing = False
        self._reads = 0
        self._writes = 0

    def retrieve_entry(self, api_type: str,
                       key: Hashable) -> Tuple[Hashable, Optional[str]]:
        with self._cond:
            while self._writes > 0 or self._is_writing:
                self._cond.wait()
            self._reads += 1

        key_pair = self._constr_key_pair(api_type, key)
        val = self._cache.get(key_pair, None)

        with self._cond:
            self._reads -= 1
            self._cond.notify_all()

        return key, val

    def upsert_entry(self,
                     api_type: str,
                     key: Hashable,
                     value: Union[str, Callable[[], str]]) -> Optional[str]:
        with self._cond:
            self._writes += 1
            while self._reads > 0 or self._is_writing:
                self._cond.wait()
            self._writes -= 1
            self._is_writing = True

        timestamp = time.time()
        key_pair = self._constr_key_pair(api_type, key)
        if timestamp - self._last_update_time.get(
                key_pair,
                -self._MIN_UPDATE_INTERVAL) > self._MIN_UPDATE_INTERVAL:
            if callable(value):
                try:
                    val = value()
                except Exception as e:
                    logger.error(
                        "An error was encountered while computing the value.",
                        exc_info=e)
                    val = None
            else:
                val = value
            if val is not None:
                self._cache[key_pair] = val
                self._last_update_time[key_pair] = time.time()
        else:
            val = self._cache[key_pair]

        with self._cond:
            self._is_writing = False
            self._cond.notify_all()

        return val

    def _constr_key_pair(self, key1: str,
                         key2: Hashable) -> Tuple[str, Hashable]:
        return (key1, key2)


class AuthManager(object):
    def __init__(self,
                 api_type: APIType,
                 *,
                 auth_token: Optional[str]=None,
                 **kwargs: Any) -> None:
        super().__init__()
        self.api_type = api_type
        self._cfg = dict(**kwargs)
        self._cache_key = self._get_cache_key()
        self._token = self._init_auth_token(auth_token)

    def get_auth_token(self) -> str:
        return self._token

    def update_auth_token(self) -> str:
        new_token = self._update_auth_token(self._token)
        self._token = new_token
        logger.info("Security token is updated.")
        return self._token

    def _request_auth_token(self, init: bool) -> str:
        raise NotImplementedError

    def _get_cache_key(self) -> Hashable:
        raise NotImplementedError

    def _init_auth_token(self, token: Optional[str]) -> str:
        if token is None:
            cached_token = self._retrieve_from_cache()
            if cached_token is not None:
                logger.info("Cached security token will be used.")
                token = cached_token
            else:
                logger.info(
                    "Security token is not set. "
                    "It will be retrieved or generated based on other parameters."
                )
                token = self._update_cache(init=True)
        return token

    def _update_auth_token(self, old_token: str) -> str:
        cached_token = self._retrieve_from_cache()
        if cached_token is not None and cached_token != old_token:
            new_token = cached_token
        else:
            new_token = self._update_cache(init=False)
        return new_token

    def _retrieve_from_cache(self) -> Optional[str]:
        return _GlobalAuthCache().retrieve_entry(self.api_type.name,
                                                 self._cache_key)[1]

    def _update_cache(self, init: bool) -> str:
        token = _GlobalAuthCache().upsert_entry(
            self.api_type.name,
            self._cache_key,
            functools.partial(
                self._request_auth_token, init=init))
        if token is None:
            raise errors.TokenUpdateFailedError
        else:
            logger.debug("Cache is updated.")
            return token


class BCEAuthManager(AuthManager):
    def __init__(self,
                 api_type: APIType,
                 *,
                 auth_token: Optional[str]=None,
                 ak: Optional[str]=None,
                 sk: Optional[str]=None,
                 **kwargs: Any) -> None:
        super().__init__(
            api_type, auth_token=auth_token, ak=ak, sk=sk, **kwargs)

    def _request_auth_token(self, init: bool) -> str:
        # `init` not used
        url = "https://aip.baidubce.com/oauth/2.0/token"
        ak = self._cfg['ak']
        sk = self._cfg['sk']
        if ak is None or sk is None:
            raise ValueError("Invalid API key or secret key")
        params = {
            'grant_type': 'client_credentials',
            'client_id': ak,
            'client_secret': sk
        }
        result = requests.request(method='GET', url=url, params=params)
        if result.status_code != http.HTTPStatus.OK:
            raise errors.HTTPRequestError(
                f"Status code is not {http.HTTPStatus.OK}.",
                rcode=result.status_code,
                rbody=result.content.decode('utf-8'),
                rheaders=result.headers)
        else:
            rbody = result.content.decode('utf-8')
            rbody = json.loads(rbody)
            if not isinstance(rbody, dict):
                raise errors.HTTPRequestError(
                    "Response body cannot be deserialized to a dict.")
            token = rbody['access_token']
            return token

    def _get_cache_key(self) -> Hashable:
        return (self._cfg['ak'], self._cfg['sk'])
