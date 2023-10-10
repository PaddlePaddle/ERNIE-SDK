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

import os
import numbers
import pathlib
import re
import sys
import types
from typing import (Any, Callable, Dict, Generic, Optional, Type, TypeVar)

from .types import (ConfigDictType)
from .utils.misc import Singleton

__all__ = ['GlobalConfig', 'init_global_config']


class BaseConfig(object):
    def __init__(self,
                 cfg_dict: Optional[Dict[str, '_ConfigItem']]=None) -> None:
        super().__init__()
        self._cfg_dict: Dict[
            str, '_ConfigItem'] = cfg_dict if cfg_dict is not None else dict()

    def add_item(self, cfg: '_ConfigItem') -> None:
        if not isinstance(cfg, _ConfigItem):
            raise TypeError
        self._cfg_dict[cfg.key] = cfg

    def get_value(self, key: str) -> Optional[Any]:
        cfg = self._cfg_dict[key]
        return cfg.value

    def set_value(self, key: str, value: Optional[Any]) -> None:
        cfg = self._cfg_dict[key]
        cfg.value = value


class GlobalConfig(BaseConfig, metaclass=Singleton):
    def create_dict(self, **overrides: Optional[Any]) -> ConfigDictType:
        dict_ = {}
        for key, cfg in self._cfg_dict.items():
            if key in overrides:
                val = overrides.pop(key)
            else:
                val = cfg.value
            if val is not None:
                cfg.validate(val)
            dict_[key] = val
        if len(overrides) != 0:
            raise KeyError(f"Unexpected keys: {list(overrides.keys())}")
        return dict_


def init_global_config() -> None:
    cfg = GlobalConfig()

    # Authentication settings
    # Access token
    cfg.add_item(StringItem(key='access_token', env_key='EB_ACCESS_TOKEN'))
    # API key or access key ID
    cfg.add_item(StringItem(key='ak', env_key='EB_AK'))
    # Secret key or secret access key
    cfg.add_item(StringItem(key='sk', env_key='EB_SK'))

    # API backend settings
    # API base URL
    cfg.add_item(URLItem(key='api_base_url', env_key='EB_BASE_URL'))
    # API type
    cfg.add_item(
        StringItem(
            key='api_type', env_key='EB_API_TYPE', default='qianfan'))

    # Miscellaneous settings
    # Proxy to use
    cfg.add_item(URLItem(key='proxy', env_key='EB_PROXY'))
    # Timeout for retrying
    cfg.add_item(PositiveNumberItem(key='timeout', env_key='EB_TIMEOUT'))


_T = TypeVar('_T')


class _ConfigItem(Generic[_T]):
    DATA_TYPE: Optional[Type[_T]]
    _FACTORY: Callable[[str], _T]

    def __init__(self,
                 key: str,
                 env_key: Optional[str]=None,
                 default: Optional[_T]=None) -> None:
        super().__init__()
        self._key: str = key
        self._env_key: Optional[str] = env_key
        self._def_val: Optional[_T] = default

        self.data_type: Optional[Type[_T]] = self.DATA_TYPE
        self._env_val: Optional[_T] = None
        if self._env_key is not None:
            env_val = os.environ.get(self._env_key, None)
            if env_val is not None:
                self._env_val = type(self)._FACTORY(env_val)
        self._val: Optional[_T] = None

    @property
    def key(self) -> str:
        return self._key

    @property
    def value(self) -> Optional[_T]:
        if self._val is not None:
            return self._val
        else:
            mod = sys.modules['erniebot']
            if self._key in mod.__dict__:
                val = mod.__dict__[self._key]
                assert not isinstance(val, types.ModuleType)
                return val
            else:
                if self._env_val is not None:
                    return self._env_val
                else:
                    return self._def_val

    @value.setter
    def value(self, new_value: Optional[_T]) -> None:
        if new_value is not None:
            self.validate(new_value)
        self._val = new_value

    def validate(self, val: _T) -> None:
        raise NotImplementedError

    def __str__(self) -> str:
        return f"{{{repr(self.key)}: {self.value}}}"


class NumberItem(_ConfigItem[float]):
    DATA_TYPE = float
    _FACTORY = float

    def validate(self, val: float) -> None:
        if not isinstance(val, numbers.Real):
            raise TypeError


class PositiveNumberItem(NumberItem):
    def validate(self, val: float) -> None:
        super().validate(val)
        if val < 0.0:
            raise ValueError(
                f"Invalid value ({val}) for {self.key}, which should be a positive value."
            )


class StringItem(_ConfigItem[str]):
    DATA_TYPE = str
    _FACTORY = str

    def validate(self, val: str) -> None:
        if not isinstance(val, str):
            raise TypeError


class PathItem(StringItem):
    def validate(self, val: str) -> None:
        super().validate(val)
        if not pathlib.Path(val).exists():
            raise ValueError(f"{val} does not exist.")


class URLItem(StringItem):
    def validate(self, val: str) -> None:
        super().validate(val)
        # Allow both HTTP and HTTPS
        pat = r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
        res = re.match(pat, val)
        if res is None:
            raise ValueError(f"Invalid URL: {val}")
