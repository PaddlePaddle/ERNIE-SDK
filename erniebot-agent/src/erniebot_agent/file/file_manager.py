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
from types import TracebackType
from typing import Any, Dict, List, Literal, Optional, Type, Union, final, overload

import anyio
from typing_extensions import Self, TypeAlias

from erniebot_agent.file import protocol
from erniebot_agent.file.base import File
from erniebot_agent.file.file_registry import FileRegistry
from erniebot_agent.file.local_file import LocalFile, create_local_file_from_path
from erniebot_agent.file.remote_file import RemoteFile, RemoteFileClient
from erniebot_agent.utils.exceptions import FileError
from erniebot_agent.utils.mixins import Closeable

logger = logging.getLogger(__name__)

FilePath: TypeAlias = Union[str, os.PathLike]


class FileManager(object):
    """
    Manages files, providing methods for creating, retrieving, and listing files.

    Attributes:
        registry(FileRegistry): The file registry.
        remote_file_client(RemoteFileClient): The remote file client.
        save_dir (Optional[FilePath]): Directory for saving local files.
        _file_registry (FileRegistry): Registry for keeping track of files.

    Methods:        
        create_file_from_path: Create a file from a specified file path.
        create_local_file_from_path: Create a local file from a file path.
        create_remote_file_from_path: Create a remote file from a file path.
        create_file_from_bytes: Create a file from bytes.
        retrieve_remote_file_by_id: Retrieve a remote file by its ID.
        look_up_file_by_id: Look up a file by its ID.
        list_remote_files: List remote files.
        _fs_create_file: Create a file in the file system.
        _fs_create_temp_dir: Create a temporary directory in the file system.
        _clean_up_temp_dir: Clean up a temporary directory.

    """
    _remote_file_client: Optional[RemoteFileClient]

    def __init__(
        self,
        remote_file_client: Optional[RemoteFileClient] = None,
        *,
        auto_register: bool = True,
        save_dir: Optional[FilePath] = None,
    ) -> None:
        """
        Initialize the FileManager object.

        Args:
            remote_file_client (Optional[RemoteFileClient]): The remote file client.
            auto_register (bool): Automatically register files in the file registry.
            save_dir (Optional[FilePath]): Directory for saving local files.

        Returns:
            None

        """
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
            self._temp_dir = self._create_temp_dir()
            self._save_dir = pathlib.Path(self._temp_dir.name)
        self._default_file_type = default_file_type
        self._prune_on_close = prune_on_close

        self._file_registry = FileRegistry()
        self._fully_managed_files: List[Union[LocalFile, RemoteFile]] = []

        self._closed = False

    @property
    def remote_file_client(self) -> RemoteFileClient:
        """
        Get the remote file client.

        Returns:
            RemoteFileClient: The remote file client.

        Raises:
            AttributeError: If no remote file client is set.

        """
        if self._remote_file_client is None:
            raise AttributeError("No remote file client is set.")
        else:
            return self._remote_file_client

    @property
    def closed(self):
        return self._closed

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]] = None,
        exc_value: Optional[BaseException] = None,
        traceback: Optional[TracebackType] = None,
    ) -> None:
        await self.close()

    @overload
    async def create_file_from_path(
        self,
        file_path: FilePath,
        *,
        file_purpose: protocol.FilePurpose = ...,
        file_metadata: Optional[Dict[str, Any]] = ...,
        file_type: Literal["local"],
    ) -> LocalFile:
        ...

    @overload
    async def create_file_from_path(
        self,
        file_path: FilePath,
        *,
        file_purpose: protocol.FilePurpose = ...,
        file_metadata: Optional[Dict[str, Any]] = ...,
        file_type: Literal["remote"],
    ) -> RemoteFile:
        ...

    @overload
    async def create_file_from_path(
        self,
        file_path: FilePath,
        *,
        file_purpose: FilePurpose = "assistants",
        file_metadata: Optional[Dict[str, Any]] = None,
        file_type: Optional[Literal["local", "remote"]] = None,
    ) -> Union[LocalFile, RemoteFile]:
        """
        Create a file from a specified file path.

        Args:
            file_path (FilePath): The path to the file.
            file_purpose (FilePurpose): The purpose or use case of the file.
            file_metadata (Optional[Dict[str, Any]]): Additional metadata associated with the file.
            file_type (Optional[Literal["local", "remote"]]): The type of file ("local" or "remote").

        Returns:
            Union[LocalFile, RemoteFile]: The created file.

        Raises:
            ValueError: If an unsupported file type is provided.

        """
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
            raise RuntimeError(f"Unsupported file type: {file_type}")
        return file

    async def create_local_file_from_path(
        self,
        file_path: FilePath,
        file_purpose: FilePurpose,
        file_metadata: Optional[Dict[str, Any]],
    ) -> LocalFile:
        """
        Create a local file from a local file path.

        Args:
            file_path (FilePath): The path to the file.
            file_purpose (FilePurpose): The purpose or use case of the file.
            file_metadata (Optional[Dict[str, Any]]): Additional metadata associated with the file.

        Returns:
            LocalFile: The created local file.

        """
        file = create_local_file_from_path(
            pathlib.Path(file_path),
            file_purpose,
            file_metadata or {},
        )
        self._file_registry.register_file(file)
        return file

    async def create_remote_file_from_path(
        self,
        file_path: FilePath,
        file_purpose: protocol.FilePurpose,
        file_metadata: Optional[Dict[str, Any]],
    ) -> RemoteFile:
        """
        Create a remote file from a file path.

        Args:
            file_path (FilePath): The path to the file.
            file_purpose (FilePurpose): The purpose or use case of the file.
            file_metadata (Optional[Dict[str, Any]]): Additional metadata associated with the file.

        Returns:
            RemoteFile: The created remote file.

        """
        file = await self._create_remote_file_from_path(
            pathlib.Path(file_path),
            file_purpose,
            file_metadata,
        )
        self._file_registry.register_file(file)
        self._fully_managed_files.append(file)
        return file

    @overload
    async def create_file_from_bytes(
        self,
        file_contents: bytes,
        filename: str,
        *,
        file_purpose: protocol.FilePurpose = ...,
        file_metadata: Optional[Dict[str, Any]] = ...,
        file_type: Literal["local"],
    ) -> LocalFile:
        ...

    @overload
    async def create_file_from_bytes(
        self,
        file_contents: bytes,
        filename: str,
        *,
        file_purpose: protocol.FilePurpose = ...,
        file_metadata: Optional[Dict[str, Any]] = ...,
        file_type: Literal["remote"],
    ) -> RemoteFile:
        ...

    @overload
    async def create_file_from_bytes(
        self,
        file_contents: bytes,
        filename: str,
        *,
        file_purpose: protocol.FilePurpose = ...,
        file_metadata: Optional[Dict[str, Any]] = ...,
        file_type: None = ...,
    ) -> Union[LocalFile, RemoteFile]:
        ...

    async def create_file_from_bytes(
        self,
        file_contents: bytes,
        filename: str,
        *,
        file_purpose: protocol.FilePurpose = "assistants",
        file_metadata: Optional[Dict[str, Any]] = None,
        file_type: Optional[Literal["local", "remote"]] = None,
    ) -> Union[LocalFile, RemoteFile]:
        """
        Create a file from bytes.

        Args:
            file_contents (bytes): The contents of the file.
            filename (str): The name of the file.
            file_purpose (FilePurpose): The purpose or use case of the file.
            file_metadata (Optional[Dict[str, Any]]): Additional metadata associated with the file.
            file_type (Optional[Literal["local", "remote"]]): The type of file ("local" or "remote").

        Returns:
            Union[LocalFile, RemoteFile]: The created file.

        """
        self.ensure_not_closed()
        if file_type is None:
            file_type = self._get_default_file_type()
        file_path = self._get_unique_file_path(
            prefix=pathlib.PurePath(filename).stem,
            suffix=pathlib.PurePath(filename).suffix,
        )
        async_file_path = anyio.Path(file_path)
        await async_file_path.touch()
        should_remove_file = True
        try:
            async with await async_file_path.open("wb") as f:
                await f.write(file_contents)
            file: Union[LocalFile, RemoteFile]
            if file_type == "local":
                file = await self._create_local_file_from_path(file_path, file_purpose, file_metadata)
                should_remove_file = False
            elif file_type == "remote":
                file = await self._create_remote_file_from_path(
                    file_path,
                    file_purpose,
                    file_metadata,
                )
            else:
                raise RuntimeError(f"Unsupported file type: {file_type}")
        finally:
            if should_remove_file:
                await async_file_path.unlink()
        self._file_registry.register_file(file)
        self._fully_managed_files.append(file)
        return file

    async def retrieve_remote_file_by_id(self, file_id: str) -> RemoteFile:
        """
        Retrieve a remote file by its ID.

        Args:
            file_id (str): The ID of the remote file.

        Returns:
            RemoteFile: The retrieved remote file.

        """
        self.ensure_not_closed()
        file = await self.remote_file_client.retrieve_file(file_id)
        self._file_registry.register_file(file)
        return file

    async def list_remote_files(self) -> List[RemoteFile]:
        self.ensure_not_closed()
        files = await self.remote_file_client.list_files()
        return files

    def look_up_file_by_id(self, file_id: str) -> Optional[File]:
        """
        Look up a file by its ID.

        Args:
            file_id (str): The ID of the file.

        Returns:
            Optional[Union[LocalFile, RemoteFile]]: The looked-up file, or None if not found.

        Raises:
            FileError: If the file with the specified ID is not found.

        """
        self.ensure_not_closed()
        file = self._file_registry.look_up_file(file_id)
        if file is None:
            raise FileError(
                f"File with ID {repr(file_id)} not found. "
                "Please check if `file_id` is correct and the file is registered."
            )
        return file

    def list_registered_files(self) -> List[File]:
        """
        List remote files.

        Returns:
            List[RemoteFile]: The list of remote files.

        """
        self.ensure_not_closed()
        return self._file_registry.list_files()

    async def prune(self) -> None:
        for file in self._fully_managed_files:
            if isinstance(file, RemoteFile):
                # FIXME: Currently this is not supported.
                # await file.delete()
                pass
            elif isinstance(file, LocalFile):
                assert self._save_dir in file.path.parents
                await anyio.Path(file.path).unlink()
            else:
                raise AssertionError("Unexpected file type")
            self._file_registry.unregister_file(file)
        self._fully_managed_files.clear()

    async def close(self) -> None:
        if not self._closed:
            if self._remote_file_client is not None:
                await self._remote_file_client.close()
            if self._prune_on_close:
                await self.prune()
            if self._temp_dir is not None:
                self._clean_up_temp_dir(self._temp_dir)
            self._closed = True

    async def _create_local_file_from_path(
        self,
        file_path: pathlib.Path,
        file_purpose: protocol.FilePurpose,
        file_metadata: Optional[Dict[str, Any]],
    ) -> LocalFile:
        return create_local_file_from_path(
            pathlib.Path(file_path),
            file_purpose,
            file_metadata or {},
        )

    async def _create_remote_file_from_path(
        self,
        file_path: pathlib.Path,
        file_purpose: protocol.FilePurpose,
        file_metadata: Optional[Dict[str, Any]],
    ) -> RemoteFile:
        file = await self.remote_file_client.upload_file(file_path, file_purpose, file_metadata or {})
        return file

    def _get_default_file_type(self) -> Literal["local", "remote"]:
        if self._default_file_type is not None:
            return self._default_file_type
        else:
            if self._remote_file_client is not None:
                return "remote"
            else:
                return "local"

    def _get_unique_file_path(
        self, prefix: Optional[str] = None, suffix: Optional[str] = None
    ) -> pathlib.Path:
        filename = f"{prefix or ''}{str(uuid.uuid4())}{suffix or ''}"
        file_path = self._save_dir / filename
        return file_path

    @staticmethod
    def _create_temp_dir() -> tempfile.TemporaryDirectory:
        temp_dir = tempfile.TemporaryDirectory()
        return temp_dir

    @staticmethod
    def _clean_up_temp_dir(temp_dir: tempfile.TemporaryDirectory) -> None:
        try:
            temp_dir.cleanup()
        except Exception as e:
            logger.warning("Failed to clean up temporary directory: %s", temp_dir.name, exc_info=e)
