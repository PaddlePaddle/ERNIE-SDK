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

import threading
from collections.abc import AsyncIterator, Iterator
from typing import ClassVar

__all__ = ["Constant", "SingletonMeta", "NOT_GIVEN", "NotGiven", "filter_args", "transform"]


class Constant(object):
    def __init__(self, val):
        super().__init__()
        self.val = val

    def __get__(self, obj, type_=None):
        return self.val

    def __set__(self, obj, val):
        raise AttributeError("The value of a constant cannot be modified.")


class SingletonMeta(type):
    _insts: ClassVar[dict] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._insts:
            with cls._lock:
                if cls not in cls._insts:
                    cls._insts[cls] = super().__call__(*args, **kwargs)
        return cls._insts[cls]


class _NotGivenSentinel(metaclass=SingletonMeta):
    def __bool__(self):
        return False

    def __repr__(self):
        return "NOT_GIVEN"


NOT_GIVEN = _NotGivenSentinel()
NotGiven = _NotGivenSentinel


def filter_args(**kwargs):
    return {k: v for k, v in kwargs.items() if v is not NOT_GIVEN}


def transform(func, data):
    if isinstance(data, Iterator):
        return (func(d) for d in data)
    elif isinstance(data, AsyncIterator):
        return (func(d) async for d in data)
    else:
        return func(data)
