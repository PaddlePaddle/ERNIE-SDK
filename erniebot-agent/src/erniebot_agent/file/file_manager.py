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

import contextlib
import contextvars
import logging
import os
import pathlib
import tempfile
import uuid
from collections import deque
from types import TracebackType
from typing import (
    Any,
    Deque,
    Dict,
    Generator,
    List,
    Literal,
    Optional,
    Type,
    Union,
    final,
    overload,
)

import anyio
from typing_extensions import Self, TypeAlias, assert_never

from erniebot_agent.file import protocol
from erniebot_agent.file.file_registry import FileRegistry
from erniebot_agent.file.local_file import LocalFile, create_local_file_from_path
from erniebot_agent.file.remote_file import RemoteFile, RemoteFileClient
from erniebot_agent.utils.exceptions import FileError
from erniebot_agent.utils.mixins import Closeable, Noncopyable

FilePath: TypeAlias = Union[str, os.PathLike]
File: TypeAlias = Union[LocalFile, RemoteFile]

_logger = logging.getLogger(__name__)

_default_file_manager_var: contextvars.ContextVar[Optional["FileManager"]] = contextvars.ContextVar(
    "_default_file_manager_var", default=None
)


def get_default_file_manager() -> Optional["FileManager"]:
    return _default_file_manager_var.get()


@final
class FileManager(Closeable, Noncopyable):
    """
    Manages files, providing methods for creating, retrieving, and listing files.

    Attributes:
        remote_file_client(RemoteFileClient): The remote file client.
        save_dir (Optional[FilePath]): Directory for saving local files.
        closed: Whether the file manager is closed.

    Methods:
        create_file_from_path: Create a file from a specified file path.
        create_local_file_from_path: Create a local file from a file path.
        create_remote_file_from_path: Create a remote file from a file path.
        create_file_from_bytes: Create a file from bytes.
        retrieve_remote_file_by_id: Retrieve a remote file by its ID.
        look_up_file_by_id: Look up a file by its ID.
        list_remote_files: List remote files.

    """

    _temp_dir: Optional[tempfile.TemporaryDirectory] = None

    def __init__(
        self,
        remote_file_client: Optional[RemoteFileClient] = None,
        save_dir: Optional[FilePath] = None,
        *,
        prune_on_close: bool = True,
    ) -> None:
        """
        Initialize the FileManager object.

        Args:
            remote_file_client (Optional[RemoteFileClient]): The remote file client.
            prune_on_close (bool): Control whether to automatically clean up files
                                   that can be safely deleted when the object is closed.
            save_dir (Optional[FilePath]): Directory for saving local files.

        Returns:
            None

        """
        super().__init__()

        self._remote_file_client = remote_file_client
        if save_dir is not None:
            self._save_dir = pathlib.Path(save_dir)
        else:
            # This can be done lazily, but we need to be careful about race conditions.
            self._temp_dir = self._create_temp_dir()
            self._save_dir = pathlib.Path(self._temp_dir.name)
        self._prune_on_close = prune_on_close

        self._file_registry: FileRegistry[File] = FileRegistry()
        self._fully_managed_files: Deque[File] = deque()

        self._closed = False

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
        file_purpose: protocol.FilePurpose = ...,
        file_metadata: Optional[Dict[str, Any]] = ...,
        file_type: None = ...,
    ) -> File:
        ...

    async def create_file_from_path(
        self,
        file_path: FilePath,
        *,
        file_purpose: protocol.FilePurpose = "assistants",
        file_metadata: Optional[Dict[str, Any]] = None,
        file_type: Optional[Literal["local", "remote"]] = None,
    ) -> File:
        """
        Create a file from a specified file path.

        Args:
            file_path (FilePath): The path to the file.
            file_purpose (FilePurpose): The purpose or use case of the file,
                    including `assistant`: used for llm and `assistant_output`: used for output.
            file_metadata (Optional[Dict[str, Any]]): Additional metadata associated with the file.
            file_type (Optional[Literal["local", "remote"]]): The type of file ("local" or "remote").

        Returns:
            Union[LocalFile, RemoteFile]: The created file.

        Raises:
            ValueError: If an unsupported file type is provided.

        """
        self.ensure_not_closed()
        file: File
        if file_type is None:
            file_type = self._get_default_file_type()
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
        file_purpose: protocol.FilePurpose,
        file_metadata: Optional[Dict[str, Any]],
    ) -> LocalFile:
        """
        Create a local file from a local file path.

        Args:
            file_path (FilePath): The path to the file.
            file_purpose (FilePurpose): The purpose or use case of the file,
                    including `assistant`: used for llm and `assistant_output`: used for output.
            file_metadata (Optional[Dict[str, Any]]): Additional metadata associated with the file.

        Returns:
            LocalFile: The created local file.

        """
        file = await self._create_local_file_from_path(
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
        Create a remote file from a file path and upload it to the client.

        Args:
            file_path (FilePath): The path to the file.
            file_purpose (FilePurpose): The purpose or use case of the file,
                    including `assistant`: used for llm and `assistant_output`: used for output.
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
    ) -> File:
        ...

    async def create_file_from_bytes(
        self,
        file_contents: bytes,
        filename: str,
        *,
        file_purpose: protocol.FilePurpose = "assistants",
        file_metadata: Optional[Dict[str, Any]] = None,
        file_type: Optional[Literal["local", "remote"]] = None,
    ) -> File:
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
            file: File
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
                raise ValueError(f"Unsupported file type: {file_type}")
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
        file = await self._get_remote_file_client().retrieve_file(file_id)
        self._file_registry.register_file(file)
        return file

    async def list_remote_files(self) -> List[RemoteFile]:
        self.ensure_not_closed()
        files = await self._get_remote_file_client().list_files()
        return files

    def look_up_file_by_id(self, file_id: str) -> File:
        """
        Look up a file by its ID.

        Args:
            file_id (str): The ID of the file.

        Returns:
            file[File]: The looked-up file.

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
        """Clean local cache of file manager."""
        while True:
            try:
                file = self._fully_managed_files.pop()
            except IndexError:
                break
            if isinstance(file, RemoteFile):
                # FIXME: Currently this is not supported.
                # await file.delete()
                pass
            elif isinstance(file, LocalFile):
                assert self._save_dir.resolve() in file.path.resolve().parents
                await anyio.Path(file.path).unlink()
            else:
                assert_never()
            self._file_registry.unregister_file(file)

    async def close(self) -> None:
        """Delete the file manager and clean up its cache"""
        if not self._closed:
            if self._remote_file_client is not None:
                await self._remote_file_client.close()
            if self._prune_on_close:
                await self.prune()
            if self._temp_dir is not None:
                self._clean_up_temp_dir(self._temp_dir)
            self._closed = True

    @contextlib.contextmanager
    def as_default_file_manager(self) -> Generator[None, None, None]:
        token = _default_file_manager_var.set(self)
        try:
            yield
        finally:
            _default_file_manager_var.reset(token)

    def sniff_and_extract_files_from_list(self, list_: List[Any]) -> List[File]:
        files: List[File] = []
        for item in list_:
            if not isinstance(item, str):
                continue
            if protocol.is_file_id(item):
                file_id = item
                try:
                    file = self.look_up_file_by_id(file_id)
                except FileError as e:
                    raise FileError(f"An unregistered file with ID {repr(file_id)} was found.") from e
                files.append(file)
        return files

    def sniff_and_extract_files_from_text(self, text: str) -> List[File]:
        file_ids = protocol.extract_file_ids(text)
        files: List[File] = []
        for file_id in file_ids:
            if protocol.is_file_id(file_id):
                try:
                    file = self.look_up_file_by_id(file_id)
                except FileError as e:
                    raise FileError(f"An unregistered file with ID {repr(file_id)} was found.") from e
                files.append(file)
        return files

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
        file = await self._get_remote_file_client().upload_file(file_path, file_purpose, file_metadata or {})
        return file

    def _get_remote_file_client(self) -> RemoteFileClient:
        if self._remote_file_client is None:
            raise RuntimeError("No remote file client is set.")
        else:
            return self._remote_file_client

    def _get_default_file_type(self) -> Literal["local", "remote"]:
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
            _logger.warning("Failed to clean up temporary directory: %s", temp_dir.name, exc_info=e)
