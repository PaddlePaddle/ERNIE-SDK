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
    """
    Manages files, providing methods for creating, retrieving, and listing files.

    Attributes:
        registry(FileRegistry): The file registry.
        remote_file_client(RemoteFileClient): The remote file client.
        save_dir (Optional[FilePath]): Directory for saving local files.
        _file_registry (FileRegistry): Registry for keeping track of files.

    Methods:
        __init__: Initialize the FileManager object.
        
        
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
            self._save_dir = self._fs_create_temp_dir()

        self._file_registry = get_file_registry()

    @property
    def registry(self) -> FileRegistry:
        """
        Get the file registry.

        Returns:
            FileRegistry: The file registry.

        """
        return self._file_registry

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
            raise ValueError(f"Unsupported file type: {file_type}")
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
        self, file_path: FilePath, file_purpose: FilePurpose, file_metadata: Optional[Dict[str, Any]]
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
        """
        Create a file from bytes.

        Args:
            file_contents (bytes): The content bytes of the file.
            filename (str): The name of the file.
            file_purpose (FilePurpose): The purpose or use case of the file.
            file_metadata (Optional[Dict[str, Any]]): Additional metadata associated with the file.
            file_type (Optional[Literal["local", "remote"]]): The type of file ("local" or "remote").

        Returns:
            Union[LocalFile, RemoteFile]: The created file.

        """
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
        """
        Retrieve a remote file by its ID.

        Args:
            file_id (str): The ID of the remote file.

        Returns:
            RemoteFile: The retrieved remote file.

        """
        file = await self.remote_file_client.retrieve_file(file_id)
        if self._auto_register:
            self._file_registry.register_file(file, allow_overwrite=True)
        return file

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
        file = self._file_registry.look_up_file(file_id)
        if file is None:
            raise FileError(
                f"File with ID '{file_id}' not found. "
                "Please check if the file exists and the `file_id` is correct."
            )
        return file

    async def list_remote_files(self) -> List[RemoteFile]:
        """
        List remote files.

        Returns:
            List[RemoteFile]: The list of remote files.

        """
        files = await self.remote_file_client.list_files()
        if self._auto_register:
            for file in files:
                self._file_registry.register_file(file, allow_overwrite=True)
        return files

    async def _fs_create_file(
        self, prefix: Optional[str] = None, suffix: Optional[str] = None
    ) -> anyio.Path:
        """Create a file in the file system."""
        filename = f"{prefix or ''}{str(uuid.uuid4())}{suffix or ''}"
        file_path = anyio.Path(self._save_dir / filename)
        await file_path.touch()
        return file_path

    def _fs_create_temp_dir(self) -> pathlib.Path:
        """Create a temporary directory in the file system."""
        temp_dir = tempfile.TemporaryDirectory()
        # The temporary directory shall be cleaned up when the file manager is
        # garbage collected.
        weakref.finalize(self, self._clean_up_temp_dir, temp_dir)
        return pathlib.Path(temp_dir.name)

    @staticmethod
    def _clean_up_temp_dir(temp_dir: tempfile.TemporaryDirectory) -> None:
        """Clean up a temporary directory."""
        try:
            temp_dir.cleanup()
        except Exception as e:
            logger.warning("Failed to clean up temporary directory: %s", temp_dir.name, exc_info=e)
