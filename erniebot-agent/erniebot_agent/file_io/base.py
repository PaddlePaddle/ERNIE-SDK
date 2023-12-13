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
from typing import Any, Dict, Optional


class File(metaclass=abc.ABCMeta):
    def __init__(
        self,
        *,
        id: str,
        filename: str,
        byte_size: int,
        created_at: int,
        purpose: str,
        metadata: Dict[str, Any],
        URL: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.id = id
        self.filename = filename
        self.byte_size = byte_size
        self.created_at = created_at
        self.purpose = purpose
        self.metadata = metadata
        self.URL = URL
        self._param_names = ["id", "filename", "byte_size", "created_at", "purpose", "metadata", "URL"]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, File):
            return self.id == other.id
        else:
            return False

    def __repr__(self) -> str:
        attrs_str = self._get_attrs_str()
        return f"<{self.__class__.__name__} {attrs_str}>"

    def file_repr_wo_URL(self) -> str:
        return f"<file>{self.id}</file>"

    def file_repr_with_URL(self) -> str:
        if self.URL is None:
            self.URL = self._get_url()
        return f"<file>{self.id}</file><url>{self.URL}</url>"

        # Other Options including, test result is not good enough:
        # f"<file>{self.id+'<split>'+self.filename}</file><url>{self.URL}</url>"
        # f"<fileid>{self.id}</fileid><file>{self.filename}</file><url>{self.URL}</url>"

    def _get_url(self) -> str:
        """Get URL from AiStudio."""
        # TODO(shiyutang): Get URL from AiStudio.
        return ""

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self._param_names}

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
                f"metadata: {repr(self.metadata)}",
                f"URL: {repr(self.URL)}",
            ]
        )
