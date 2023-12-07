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

import abc
from typing import Any, Dict


class File(metaclass=abc.ABCMeta):
    def __init__(
        self,
        *,
        id: str,
        filename: str,
        byte_size: int,
        created_at: int,
        purpose: str,
        meta: Dict[str, Any],
    ) -> None:
        super().__init__()
        self.id = id
        self.filename = filename
        self.byte_size = byte_size
        self.created_at = created_at
        self.purpose = purpose
        self.meta = meta

    def __eq__(self, other: object) -> bool:
        if isinstance(other, File):
            return self.id == other.id
        else:
            return False

    def __repr__(self) -> str:
        attrs_str = self._get_attrs_str()
        return f"<{self.__class__.__name__} {attrs_str}>"

    @abc.abstractmethod
    async def read_contents(self) -> bytes:
        raise NotImplementedError

    def _get_attrs_str(self) -> str:
        return ", ".join(
            [
                f"id: {repr(self.id)}",
                f"filename: {repr(self.filename)}",
                f"byte_size: {repr(self.byte_size)}",
                f"created_at: {repr(self.created_at)}",
                f"purpose: {repr(self.purpose)}",
                f"meta: {repr(self.meta)}",
            ]
        )
