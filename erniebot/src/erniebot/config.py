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

import numbers
import os
import pathlib
import re
import sys
import types
from typing import Any, Dict, Optional

from .errors import ConfigItemNotFoundError
from .types import ConfigDictType
from .utils.misc import SingletonMeta

__all__ = ["GlobalConfig", "init_global_config"]


def init_global_config() -> None:
    cfg = GlobalConfig()

    # API backend settings
    # API type
    cfg.add_item(StringItem(key="api_type", env_key="EB_API_TYPE", default="qianfan"))
    # API base URL
    cfg.add_item(URLItem(key="api_base_url", env_key="EB_BASE_URL"))
    # Access token
    cfg.add_item(StringItem(key="access_token", env_key="EB_ACCESS_TOKEN"))
    # API key or access key ID
    cfg.add_item(StringItem(key="ak", env_key="EB_AK"))
    # Secret key or secret access key
    cfg.add_item(StringItem(key="sk", env_key="EB_SK"))

    # Retrying settings
    # Maximum number of retries
    cfg.add_item(
        PositiveNumberItem(key="max_retries", env_key="EB_MAX_RETRIES", default=0, ensure_integer=True)
    )
    # Minimum retry delay
    cfg.add_item(PositiveNumberItem(key="min_retry_delay", env_key="EB_MIN_RETRY_DELAY", default=1))
    # Maximum retry delay (not taking account of jitter)
    cfg.add_item(PositiveNumberItem(key="max_retry_delay", env_key="EB_MAX_RETRY_DELAY", default=10))

    # Miscellaneous settings
    # Proxy to use
    cfg.add_item(URLItem(key="proxy", env_key="EB_PROXY"))
    # requests session
    cfg.add_item(AnyObjectItem(key="requests_session"))
    # aiohttp session
    cfg.add_item(AnyObjectItem(key="aiohttp_session"))


class _Config(object):
    def __init__(self, cfg_dict: Optional[Dict[str, "_ConfigItem"]] = None) -> None:
        super().__init__()
        self._cfg_dict: Dict[str, "_ConfigItem"] = cfg_dict if cfg_dict is not None else dict()

    def add_item(self, cfg: "_ConfigItem") -> None:
        if not isinstance(cfg, _ConfigItem):
            raise TypeError
        self._cfg_dict[cfg.key] = cfg

    def get_value(self, key: str) -> Any:
        try:
            cfg = self._cfg_dict[key]
        except KeyError as e:
            raise ConfigItemNotFoundError from e
        return cfg.value

    def set_value(self, key: str, value: Any) -> None:
        try:
            cfg = self._cfg_dict[key]
        except KeyError as e:
            raise ConfigItemNotFoundError from e
        cfg.value = value


class GlobalConfig(_Config, metaclass=SingletonMeta):
    def create_dict(self, **overrides: Any) -> ConfigDictType:
        dict_: ConfigDictType = {}
        for key, cfg in self._cfg_dict.items():
            if key in overrides:
                val = overrides.pop(key)
                cfg.validate(val)
            else:
                val = cfg.value
            dict_[key] = val
        if len(overrides) != 0:
            raise TypeError(f"Unexpected keys: {list(overrides.keys())}")
        return dict_


class _ConfigItem(object):
    def __init__(self, key: str, env_key: Optional[str] = None, default: Any = None) -> None:
        super().__init__()
        self._key = key
        self._env_key = env_key
        self._def_val = default

        self._env_val: Optional[str] = None
        if self._env_key is not None:
            env_val = os.environ.get(self._env_key, None)
            self._env_val = env_val
        self._val: Any = None

    @property
    def key(self) -> str:
        return self._key

    @property
    def value(self) -> Any:
        if self._val is not None:
            return self._val
        else:
            mod = sys.modules["erniebot"]
            if self._key in mod.__dict__:
                val = mod.__dict__[self._key]
                assert not isinstance(val, types.ModuleType)
            else:
                if self._env_val is not None:
                    val = self.factory(self._env_val)
                else:
                    val = self._def_val
            self.validate(val)
            return val

    @value.setter
    def value(self, new_value: Any) -> None:
        self.validate(new_value)
        self._val = new_value

    def factory(self, env_val: str) -> Any:
        raise NotImplementedError

    def validate(self, val: Any) -> None:
        if val is not None:
            self._validate(val)

    def _validate(self, val: Any) -> None:
        raise NotImplementedError

    def __str__(self) -> str:
        return f"{{{repr(self.key)}: {self.value}}}"


class NumberItem(_ConfigItem):
    def __init__(
        self, key: str, env_key: Optional[str] = None, default: Any = None, *, ensure_integer: bool = False
    ) -> None:
        super().__init__(key=key, env_key=env_key, default=default)
        self.ensure_integer = ensure_integer

    def factory(self, env_val: str) -> Any:
        if self.ensure_integer:
            return int(env_val)
        else:
            return float(env_val)

    def _validate(self, val: Any) -> None:
        if not isinstance(val, int if self.ensure_integer else numbers.Real):
            raise TypeError


class PositiveNumberItem(NumberItem):
    def _validate(self, val: Any) -> None:
        super()._validate(val)
        if val < 0.0:
            raise ValueError(f"Invalid value ({val}) for {self.key}, which should be a positive value.")


class StringItem(_ConfigItem):
    def factory(self, env_val: str) -> Any:
        return str(env_val)

    def _validate(self, val: Any) -> None:
        if not isinstance(val, str):
            raise TypeError


class PathItem(StringItem):
    def _validate(self, val: Any) -> None:
        super()._validate(val)
        if not pathlib.Path(val).exists():
            raise ValueError(f"{val} does not exist.")


class URLItem(StringItem):
    def _validate(self, val: Any) -> None:
        super()._validate(val)
        # Allow both HTTP and HTTPS
        pat = r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"  # noqa: E501
        res = re.match(pat, val)
        if res is None:
            raise ValueError(f"Invalid URL: {val}")


class AnyObjectItem(_ConfigItem):
    def __init__(self, key: str, default: Any = None) -> None:
        super().__init__(key=key, env_key=None, default=default)

    def factory(self, env_val: str) -> Any:
        raise AssertionError

    def _validate(self, val: Any) -> None:
        # Any object is valid
        pass
