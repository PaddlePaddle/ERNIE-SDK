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
import pathlib
import uuid
from typing import Literal, Optional, Union, overload

import anyio
from erniebot_agent.file_io.file_registry import FileRegistry, get_file_registry
from erniebot_agent.file_io.local_file import LocalFile, create_local_file_from_path
from erniebot_agent.file_io.remote_file import RemoteFile, RemoteFileClient
from erniebot_agent.utils.temp_file import create_tracked_temp_dir
from typing_extensions import TypeAlias

_PathType: TypeAlias = Union[str, os.PathLike]


class FileManager(object):
    _remote_file_client: Optional[RemoteFileClient]

    def __init__(
        self,
        remote_file_client: Optional[RemoteFileClient] = None,
        *,
        auto_register: bool = True,
    ) -> None:
        super().__init__()
        if remote_file_client is not None:
            self._remote_file_client = remote_file_client
        else:
            self._remote_file_client = None
        self._auto_register = auto_register

        self._file_registry = get_file_registry()
        # This can be done lazily, but we need to be careful about race conditions.
        self._temp_dir = create_tracked_temp_dir()

    @property
    def registry(self) -> FileRegistry:
        return self._file_registry

    @property
    def remote_file_client(self) -> RemoteFileClient:
        if self._remote_file_client is None:
            raise AttributeError("No remote file client is set.")
        else:
            return self._remote_file_client

    @overload
    async def create_file_from_path(
        self, file_path: _PathType, *, file_type: Literal["local"] = ...
    ) -> LocalFile:
        ...

    @overload
    async def create_file_from_path(
        self, file_path: _PathType, *, file_type: Literal["remote"]
    ) -> RemoteFile:
        ...

    async def create_file_from_path(
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
        file = create_local_file_from_path(pathlib.Path(file_path))
        self._file_registry.register_file(file)
        return file

    async def create_remote_file_from_path(self, file_path: _PathType) -> RemoteFile:
        file = await self.remote_file_client.upload_file(pathlib.Path(file_path))
        if self._auto_register:
            self._file_registry.register_file(file)
        return file

    @overload
    async def create_file_from_bytes(
        self, file_contents: bytes, *, file_type: Literal["local"] = ...
    ) -> LocalFile:
        ...

    @overload
    async def create_file_from_bytes(
        self, file_contents: bytes, *, file_type: Literal["remote"]
    ) -> RemoteFile:
        ...

    async def create_file_from_bytes(
        self, file_contents: bytes, *, file_type: Literal["local", "remote"] = "local"
    ) -> Union[LocalFile, RemoteFile]:
        # Can we do this without creating a temp file?
        # For example, can we use in-memory files?
        file_path = self._create_temp_file()
        async with await anyio.open_file(file_path, "wb") as f:
            await f.write(file_contents)
        file = await self.create_file_from_path(file_path, file_type=file_type)
        return file

    async def retrieve_remote_file_by_id(self, file_id: str) -> RemoteFile:
        file = await self.remote_file_client.retrieve_file(file_id)
        if self._auto_register:
            self._file_registry.register_file(file)
        return file

    def _create_temp_file(self) -> pathlib.Path:
        filename = str(uuid.uuid4())
        file_path = self._temp_dir / filename
        file_path.touch()
        return file_path
