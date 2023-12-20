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

import functools
import logging
import os
import pathlib
import tempfile
import uuid
from types import TracebackType
from typing import Any, Dict, List, Literal, Optional, Type, Union, final, overload

import anyio
from typing_extensions import Self, TypeAlias

from erniebot_agent.file_io import protocol
from erniebot_agent.file_io.base import File
from erniebot_agent.file_io.caching import (
    FileCacheManager,
    RemoteFileWithCache,
    bind_cache_to_remote_file,
    create_default_file_cache_manager,
)
from erniebot_agent.file_io.file_registry import FileRegistry
from erniebot_agent.file_io.local_file import LocalFile, create_local_file_from_path
from erniebot_agent.file_io.remote_file import RemoteFile, RemoteFileClient
from erniebot_agent.utils.exception import FileError
from erniebot_agent.utils.mixins import Closeable

logger = logging.getLogger(__name__)

FilePath: TypeAlias = Union[str, os.PathLike]


@final
class FileManager(Closeable):
    _file_cache_manager: Optional[FileCacheManager]
    _temp_dir: Optional[tempfile.TemporaryDirectory] = None

    def __init__(
        self,
        remote_file_client: Optional[RemoteFileClient] = None,
        *,
        auto_register: bool = True,
        save_dir: Optional[FilePath] = None,
        cache_remote_files: bool = True,
    ) -> None:
        super().__init__()

        self._remote_file_client = remote_file_client
        self._auto_register = auto_register
        if save_dir is not None:
            self._save_dir = pathlib.Path(save_dir)
        else:
            # This can be done lazily, but we need to be careful about race conditions.
            self._temp_dir = self._create_temp_dir()
            self._save_dir = pathlib.Path(self._temp_dir.name)
        self._cache_remote_files = cache_remote_files

        self._file_registry = FileRegistry()
        if self._cache_remote_files:
            self._file_cache_manager = create_default_file_cache_manager()
        else:
            self._file_cache_manager = None

        self._closed = False
        self._clean_up_cache_files_on_discard = True

    @property
    def remote_file_client(self) -> RemoteFileClient:
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
        file_purpose: protocol.FilePurpose = ...,
        file_metadata: Optional[Dict[str, Any]] = ...,
        file_type: None = ...,
    ) -> Union[LocalFile, RemoteFile]:
        ...

    async def create_file_from_path(
        self,
        file_path: FilePath,
        *,
        file_purpose: protocol.FilePurpose = "assistants",
        file_metadata: Optional[Dict[str, Any]] = None,
        file_type: Optional[Literal["local", "remote"]] = None,
    ) -> Union[LocalFile, RemoteFile]:
        self.ensure_not_closed()
        file: Union[LocalFile, RemoteFile]
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
        file = create_local_file_from_path(
            pathlib.Path(file_path),
            file_purpose,
            file_metadata or {},
        )
        self.register_file(file)
        return file

    async def create_remote_file_from_path(
        self,
        file_path: FilePath,
        file_purpose: protocol.FilePurpose,
        file_metadata: Optional[Dict[str, Any]],
    ) -> RemoteFile:
        return await self._create_remote_file_from_path(
            pathlib.Path(file_path),
            file_purpose,
            file_metadata,
        )

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
                file = await self.create_local_file_from_path(file_path, file_purpose, file_metadata)
                should_remove_file = False
            elif file_type == "remote":
                file = await self._create_remote_file_from_path(
                    file_path,
                    file_purpose,
                    file_metadata,
                    cache_path=file_path if self._cache_remote_files else None,
                    init_cache_in_sync=True,
                )
                if self._cache_remote_files:
                    if isinstance(file, RemoteFileWithCache):
                        cache_path = file.cache_path
                        if cache_path is not None and await async_file_path.samefile(cache_path):
                            should_remove_file = False
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        finally:
            if should_remove_file:
                await async_file_path.unlink()
        return file

    async def retrieve_remote_file_by_id(self, file_id: str) -> RemoteFile:
        self.ensure_not_closed()
        file = await self.remote_file_client.retrieve_file(file_id)
        if self._cache_remote_files:
            file = await self._cache_remote_file(
                file,
                cache_path=None,
                init_cache_in_sync=None,
            )
        if self._auto_register:
            self.register_file(file, allow_overwrite=True)
        return file

    async def list_remote_files(self) -> List[RemoteFile]:
        self.ensure_not_closed()
        files = await self.remote_file_client.list_files()
        return files

    def register_file(
        self, file: Union[LocalFile, RemoteFile], *, allow_overwrite: bool = False, check_type: bool = True
    ) -> None:
        self.ensure_not_closed()
        self._file_registry.register_file(file, allow_overwrite=allow_overwrite, check_type=check_type)

    def unregister_file(self, file: Union[LocalFile, RemoteFile]) -> None:
        self.ensure_not_closed()
        self._file_registry.unregister_file(file)

    def look_up_file_by_id(self, file_id: str) -> Optional[File]:
        self.ensure_not_closed()
        file = self._file_registry.look_up_file(file_id)
        if file is None:
            raise FileError(
                f"File with ID '{file_id}' not found. "
                "Please check if `file_id` is correct and the file is registered."
            )
        return file

    def list_registered_files(self) -> List[File]:
        self.ensure_not_closed()
        return self._file_registry.list_files()

    async def close(self) -> None:
        if not self._closed:
            if self._file_cache_manager is not None:
                await self._file_cache_manager.close()
            if self._temp_dir is not None:
                self._clean_up_temp_dir(self._temp_dir)
            self._closed = True

    async def _create_remote_file_from_path(
        self,
        file_path: pathlib.Path,
        file_purpose: protocol.FilePurpose,
        file_metadata: Optional[Dict[str, Any]],
        *,
        cache_path: Optional[pathlib.Path] = None,
        init_cache_in_sync: Optional[bool] = None,
    ) -> RemoteFile:
        file = await self.remote_file_client.upload_file(file_path, file_purpose, file_metadata or {})
        if self._cache_remote_files:
            file = await self._cache_remote_file(
                file, cache_path=cache_path, init_cache_in_sync=init_cache_in_sync
            )
        if self._auto_register:
            self.register_file(file)
        return file

    async def _cache_remote_file(
        self,
        file: RemoteFile,
        *,
        cache_path: Optional[pathlib.Path],
        init_cache_in_sync: Optional[bool],
    ) -> RemoteFileWithCache:
        def _remove_cache_file(cache_path: pathlib.Path, logger: logging.Logger) -> None:
            try:
                cache_path.unlink(missing_ok=True)
            except Exception as e:
                logger.warning("Failed to remove cache file: %s", cache_path, exc_info=e)

        if self._file_cache_manager is None:
            raise RuntimeError("Chaching is not enabled.")
        if cache_path is None:
            cache_path = self._get_unique_file_path()
            init_cache_in_sync = None
        if not cache_path.exists():
            await anyio.Path(cache_path).touch()
        cache, _ = await self._file_cache_manager.get_or_create_cache(
            file.id,
            cache_path,
            discard_callback=functools.partial(_remove_cache_file, pathlib.Path(cache_path), logger)
            if self._clean_up_cache_files_on_discard
            else None,
            init_cache_in_sync=init_cache_in_sync,
        )
        return bind_cache_to_remote_file(cache, file)

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
            logger.warning("Failed to clean up temporary directory: %s", temp_dir.name, exc_info=e)
