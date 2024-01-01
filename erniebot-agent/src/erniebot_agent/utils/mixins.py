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

from __future__ import annotations

import warnings
from typing import Any, NoReturn, Protocol, final

from erniebot_agent.utils.exceptions import ObjectClosedError


class Closeable(Protocol):
    @property
    def closed(self) -> bool:
        ...

    def __del__(self, _warn=warnings.warn) -> None:
        if not self.closed:
            _warn(f"Unclosed object: {repr(self)}", ResourceWarning, source=self)

    async def close(self) -> None:
        ...

    def ensure_not_closed(self) -> None:
        if self.closed:
            raise ObjectClosedError(f"{repr(self)} is closed.")


class Noncopyable(object):
    @final
    def __copy__(self) -> NoReturn:
        raise TypeError("Cannot copy an instance of a non-copyable class.")

    @final
    def __deepcopy__(self, memo: Any) -> NoReturn:
        raise TypeError("Cannot deep-copy an instance of a non-copyable class.")

    @final
    def __reduce__(self) -> NoReturn:
        raise TypeError("Cannot pickle an instance of a non-copyable class.")

    @final
    def __reduce_ex__(self, protocol: Any) -> NoReturn:
        raise TypeError("Cannot pickle an instance of a non-copyable class.")
