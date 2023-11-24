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
from typing import List, Literal, Optional, Union, overload

from erniebot_agent.file_io.file_registry import FileRegistry
from erniebot_agent.file_io.local_file import LocalFile, create_local_file_from_path
from erniebot_agent.file_io.remote_file import (
    RemoteFile,
    create_remote_file_from_path,
    list_remote_files,
    retrieve_remote_file_by_id,
)
from erniebot_agent.file_io.remote_file_clients.base import RemoteFileClient
from typing_extensions import TypeAlias

_PathType: TypeAlias = Union[str, os.PathLike]


class FileManager(object):
    _remote_file_client: Optional[RemoteFileClient]

    def __init__(
        self, auto_register: bool = True, remote_file_client: Optional[RemoteFileClient] = None
    ) -> None:
        super().__init__()
        self._auto_register = auto_register
        if remote_file_client is not None:
            self._remote_file_client = remote_file_client
        else:
            self._remote_file_client = None
        self._file_registry = FileRegistry()

    @property
    def registry(self) -> FileRegistry:
        return self._file_registry

    @property
    def remote_file_client(self) -> RemoteFileClient:
        if self._remote_file_client is None:
            raise RuntimeError("No remote file client is set.")
        else:
            return self._remote_file_client

    @overload
    async def create_file(self, file_path: _PathType, *, file_type: Literal["local"] = ...) -> LocalFile:
        ...

    @overload
    async def create_file(self, file_path: _PathType, *, file_type: Literal["remote"]) -> RemoteFile:
        ...

    @overload
    async def create_file(
        self, file_path: _PathType, *, file_type: Literal["local", "remote"] = ...
    ) -> Union[LocalFile, RemoteFile]:
        ...

    async def create_file(
        self, file_path: _PathType, *, file_type: Literal["local", "remote"] = "local"
    ) -> Union[LocalFile, RemoteFile]:
        file: Union[LocalFile, RemoteFile]
        if file_type == "local":
            file = await self.create_local_file_from_path(file_path)
        elif file_type == "remote":
            file = await self.create_remote_file_from_path(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        return file

    async def create_local_file_from_path(self, file_path: _PathType) -> LocalFile:
        file = create_local_file_from_path(file_path)
        if self._auto_register:
            self._file_registry.register_file(file)
        return file

    async def create_remote_file_from_path(self, file_path: _PathType) -> RemoteFile:
        file = await create_remote_file_from_path(file_path, self.remote_file_client)
        if self._auto_register:
            self._file_registry.register_file(file)
        return file

    async def create_remote_file_from_id(self, file_id: str) -> RemoteFile:
        file = await retrieve_remote_file_by_id(file_id, self.remote_file_client)
        if self._auto_register:
            self._file_registry.register_file(file)
        return file

    async def list_remote_files(self) -> List[RemoteFile]:
        return await list_remote_files(self.remote_file_client)
