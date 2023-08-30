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

import copy
import json
from collections.abc import Mapping
from typing import (Any, Dict, Iterator, Union)

from .utils.misc import Constant


class EBResponse(Mapping):
    """A class that encapsulates an HTTP response for more convenient access of
    response body fields.

    An `EBResponse` object behaves like a read-only dictionary, except that the
    status code, response body, response headers, and the fields of the response
    body are accessible through attributes.
    """

    __slots__ = ('__dict', )

    _INNER_DICT_TYPE = Constant(dict)
    _RESERVED_KEYS = Constant(('code', 'body', 'headers'))

    def __init__(self,
                 code: int,
                 body: Union[str, Dict[str, Any]],
                 headers: Dict[str, Any]) -> None:
        """Initialize the instance based on response code, body, and headers.

        Args:
            code: Response status code.
            body: Response body. If `body` is a dictionary, the key-value pairs
                in the dictionary will also get registered, so that they can be
                accessed from the object using dot notation.
            headers: Response headers.
        """
        super().__init__()
        # Private name mangling to avoid conflicts with keys in `body`.
        self.__dict = self._INNER_DICT_TYPE(
            code=code, body=body, headers=headers)
        if isinstance(body, dict):
            self._update_from_dict(body)

    def __getitem__(self, key: str) -> Any:
        if key in self.__dict:
            return self.__dict[key]
        if hasattr(self.__class__, '__missing__'):
            return self.__class__.__missing__(self, key)
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.__dict)

    def __len__(self) -> int:
        return len(self.__dict)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(code={repr(self.code)}, body={repr(self.body)}, headers={repr(self.headers)})"

    def __str__(self) -> str:
        def _format(obj: object, level: int=0) -> str:
            INDENT = 2
            LEADING_SPACES = ' ' * INDENT * level
            LEADING_SPACES_FOR_NEXT_LEVEL = LEADING_SPACES + ' ' * INDENT
            if isinstance(obj, (dict, EBResponse)):
                items = []
                keys_to_ignore = []
                if isinstance(obj, EBResponse):
                    items.append(('code', _format(self.code, level=level + 1)))
                    if not isinstance(self.body, dict):
                        items.append(('body', _format(
                            self.body, level=level + 1)))
                    items.append(('headers', _format(
                        self.headers, level=level + 1)))
                    keys_to_ignore.extend(['code', 'body', 'headers'])
                for k, v in obj.items():
                    if k in keys_to_ignore:
                        continue
                    items.append((k, _format(v, level=level + 1)))
                s = ',\n'.join(
                    map(lambda item: f"{LEADING_SPACES_FOR_NEXT_LEVEL}{repr(item[0])}: {item[1]}",
                        items))
                return f"{{\n{s}\n{LEADING_SPACES}}}"
            elif isinstance(obj, (list, tuple)):
                if isinstance(obj, list):
                    left = '['
                    right = ']'
                else:
                    left = '('
                    right = ')'
                if len(obj) < 5:
                    s = ',\n'.join(LEADING_SPACES_FOR_NEXT_LEVEL + _format(
                        item, level=level + 1) for item in obj)
                    return f"{left}\n{s}\n{LEADING_SPACES}{right}"
                else:
                    s = ', '.join(
                        _format(
                            item, level=level + 1) for item in obj)
                    return f"{left}{s}{right}"
            else:
                return repr(obj)

        return _format(self)

    def __getattr__(self, key: str) -> Any:
        try:
            val = self.__dict[key]
            return val
        except KeyError:
            raise AttributeError

    def __reduce__(self) -> tuple:
        state = copy.copy(self.__dict)
        code = state.pop('code')
        body = state.pop('body')
        headers = state.pop('headers')
        return (self.__class__, (code, body, headers), state)

    def __setstate__(self, state: dict) -> None:
        self.__dict.update(state)

    def to_dict(self, deep_copy: bool=False) -> Dict[str, Any]:
        if deep_copy:
            return copy.deepcopy(self.__dict)
        else:
            return copy.copy(self.__dict)

    def to_json(self) -> str:
        return json.dumps(self.__dict)

    def _update_from_dict(self, dict_: Dict[str, Any]) -> None:
        for k, v in dict_.items():
            if k not in self._RESERVED_KEYS:
                self.__dict[k] = v
            else:
                raise ValueError(f"{repr(k)} is a reserved key.")
