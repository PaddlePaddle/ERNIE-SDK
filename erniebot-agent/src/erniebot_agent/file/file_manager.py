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

import logging
import os
import pathlib
import tempfile
import uuid
import weakref
from typing import Any, Dict, List, Literal, Optional, Union, overload

import anyio
from typing_extensions import TypeAlias

from erniebot_agent.file.base import File
from erniebot_agent.file.file_registry import FileRegistry, get_file_registry
from erniebot_agent.file.local_file import LocalFile, create_local_file_from_path
from erniebot_agent.file.protocol import FilePurpose
from erniebot_agent.file.remote_file import RemoteFile, RemoteFileClient
from erniebot_agent.utils.exception import FileError

logger = logging.getLogger(__name__)

FilePath: TypeAlias = Union[str, os.PathLike]


class FileManager(object):
    _remote_file_client: Optional[RemoteFileClient]

    def __init__(
        self,
        remote_file_client: Optional[RemoteFileClient] = None,
        *,
        auto_register: bool = True,
        save_dir: Optional[FilePath] = None,
    ) -> None:
        super().__init__()
        if remote_file_client is not None:
            self._remote_file_client = remote_file_client
        else:
            self._remote_file_client = None
        self._auto_register = auto_register
        if save_dir is not None:
            self._save_dir = pathlib.Path(save_dir)
        else:
            # This can be done lazily, but we need to be careful about race conditions.
            self._save_dir = self._fs_create_temp_dir()

        self._file_registry = get_file_registry()

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
        self,
        file_path: FilePath,
        *,
        file_purpose: FilePurpose = ...,
        file_metadata: Optional[Dict[str, Any]] = ...,
        file_type: Literal["local"] = ...,
    ) -> LocalFile:
        ...

    @overload
    async def create_file_from_path(
        self,
        file_path: FilePath,
        *,
        file_purpose: FilePurpose = ...,
        file_metadata: Optional[Dict[str, Any]] = ...,
        file_type: Literal["remote"],
    ) -> RemoteFile:
        ...

    async def create_file_from_path(
        self,
        file_path: FilePath,
        *,
        file_purpose: FilePurpose = "assistants",
        file_metadata: Optional[Dict[str, Any]] = None,
        file_type: Optional[Literal["local", "remote"]] = None,
    ) -> Union[LocalFile, RemoteFile]:
        file: Union[LocalFile, RemoteFile]
        if file_type is None:
            if self._remote_file_client is not None:
                file_type = "remote"
            else:
                file_type = "local"
        if file_type == "local":
            file = await self.create_local_file_from_path(file_path, file_purpose, file_metadata)
        elif file_type == "remote":
            file = await self.create_remote_file_from_path(file_path, file_purpose, file_metadata)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        return file

    async def create_local_file_from_path(
        self,
        file_path: FilePath,
        file_purpose: FilePurpose,
        file_metadata: Optional[Dict[str, Any]],
    ) -> LocalFile:
        file = create_local_file_from_path(
            pathlib.Path(file_path),
            file_purpose,
            file_metadata or {},
        )
        self._file_registry.register_file(file)
        return file

    async def create_remote_file_from_path(
        self, file_path: FilePath, file_purpose: FilePurpose, file_metadata: Optional[Dict[str, Any]]
    ) -> RemoteFile:
        file = await self.remote_file_client.upload_file(
            pathlib.Path(file_path), file_purpose, file_metadata or {}
        )
        if self._auto_register:
            self._file_registry.register_file(file)
        return file

    @overload
    async def create_file_from_bytes(
        self,
        file_contents: bytes,
        filename: str,
        *,
        file_purpose: FilePurpose = ...,
        file_metadata: Optional[Dict[str, Any]] = ...,
        file_type: Literal["local"] = ...,
    ) -> LocalFile:
        ...

    @overload
    async def create_file_from_bytes(
        self,
        file_contents: bytes,
        filename: str,
        *,
        file_purpose: FilePurpose = ...,
        file_metadata: Optional[Dict[str, Any]] = ...,
        file_type: Literal["remote"],
    ) -> RemoteFile:
        ...

    async def create_file_from_bytes(
        self,
        file_contents: bytes,
        filename: str,
        *,
        file_purpose: FilePurpose = "assistants",
        file_metadata: Optional[Dict[str, Any]] = None,
        file_type: Optional[Literal["local", "remote"]] = None,
    ) -> Union[LocalFile, RemoteFile]:
        # Can we do this with in-memory files?
        file_path = await self._fs_create_file(
            prefix=pathlib.PurePath(filename).stem, suffix=pathlib.PurePath(filename).suffix
        )
        try:
            async with await file_path.open("wb") as f:
                await f.write(file_contents)
            if file_type is None:
                if self._remote_file_client is not None:
                    file_type = "remote"
                else:
                    file_type = "local"
            file = await self.create_file_from_path(
                file_path,
                file_purpose=file_purpose,
                file_metadata=file_metadata,
                file_type=file_type,
            )
        finally:
            if file_type == "remote":
                await file_path.unlink()
        return file

    async def retrieve_remote_file_by_id(self, file_id: str) -> RemoteFile:
        file = await self.remote_file_client.retrieve_file(file_id)
        if self._auto_register:
            self._file_registry.register_file(file, allow_overwrite=True)
        return file

    def look_up_file_by_id(self, file_id: str) -> Optional[File]:
        file = self._file_registry.look_up_file(file_id)
        if file is None:
            raise FileError(
                f"File with ID '{file_id}' not found. "
                "Please check if the file exists and the `file_id` is correct."
            )
        return file

    async def list_remote_files(self) -> List[RemoteFile]:
        files = await self.remote_file_client.list_files()
        if self._auto_register:
            for file in files:
                self._file_registry.register_file(file, allow_overwrite=True)
        return files

    async def _fs_create_file(
        self, prefix: Optional[str] = None, suffix: Optional[str] = None
    ) -> anyio.Path:
        filename = f"{prefix or ''}{str(uuid.uuid4())}{suffix or ''}"
        file_path = anyio.Path(self._save_dir / filename)
        await file_path.touch()
        return file_path

    def _fs_create_temp_dir(self) -> pathlib.Path:
        temp_dir = tempfile.TemporaryDirectory()
        # The temporary directory shall be cleaned up when the file manager is
        # garbage collected.
        weakref.finalize(self, self._clean_up_temp_dir, temp_dir)
        return pathlib.Path(temp_dir.name)

    @staticmethod
    def _clean_up_temp_dir(temp_dir: tempfile.TemporaryDirectory) -> None:
        try:
            temp_dir.cleanup()
        except Exception as e:
            logger.warning("Failed to clean up temporary directory: %s", temp_dir.name, exc_info=e)
