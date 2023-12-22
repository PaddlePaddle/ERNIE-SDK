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
import os
from typing import Any, Dict, Union

import anyio


class File(metaclass=abc.ABCMeta):
    """
    Abstract base class representing a generic file.

    Attributes:
        id (str): Unique identifier for the file.
        filename (str): File name.
        byte_size (int): Size of the file in bytes.
        created_at (str): Timestamp indicating the file creation time.
        purpose (str): Purpose or use case of the file. []
        metadata (Dict[str, Any]): Additional metadata associated with the file.

    Methods:
        read_contents: Abstract method to asynchronously read the file contents.
        write_contents_to: Asynchronously write the file contents to a local path.
        get_file_repr: Return a string representation for use in specific contexts.
        to_dict: Convert the File object to a dictionary.
    """

    def __init__(
        self,
        *,
        id: str,
        filename: str,
        byte_size: int,
        created_at: str,
        purpose: str,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Init method for the File class.

        Args:
            id (str): Unique identifier for the file.
            filename (str): File name.
            byte_size (int): Size of the file in bytes.
            created_at (str): Timestamp indicating the file creation time.
            purpose (str): Purpose or use case of the file. []
            metadata (Dict[str, Any]): Additional metadata associated with the file.

        Returns:
            None
        """
        super().__init__()
        self.id = id
        self.filename = filename
        self.byte_size = byte_size
        self.created_at = created_at
        self.purpose = purpose
        self.metadata = metadata
        self._param_names = ["id", "filename", "byte_size", "created_at", "purpose", "metadata"]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, File):
            return self.id == other.id
        else:
            return NotImplemented

    def __repr__(self) -> str:
        attrs_str = self._get_attrs_str()
        return f"<{self.__class__.__name__} {attrs_str}>"

    @abc.abstractmethod
    async def read_contents(self) -> bytes:
        raise NotImplementedError

    async def write_contents_to(self, local_path: Union[str, os.PathLike]) -> None:
        contents = await self.read_contents()
        await anyio.Path(local_path).write_bytes(contents)

    def get_file_repr(self) -> str:
        return f"<file>{self.id}</file>"

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self._param_names}

    def _get_attrs_str(self) -> str:
        return ", ".join(
            [
                f"id: {repr(self.id)}",
                f"filename: {repr(self.filename)}",
                f"byte_size: {repr(self.byte_size)}",
                f"created_at: {repr(self.created_at)}",
                f"purpose: {repr(self.purpose)}",
                f"metadata: {repr(self.metadata)}",
            ]
        )
