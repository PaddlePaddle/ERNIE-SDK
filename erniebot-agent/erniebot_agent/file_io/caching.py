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

import asyncio
import pathlib
import weakref
from typing import Any, Awaitable, Callable, NoReturn, Optional, Tuple, final

import anyio
from erniebot_agent.file_io.remote_file import RemoteFile
from erniebot_agent.utils.logging import logger
from typing_extensions import Self, TypeAlias

_DEFAULT_CACHE_TIMEOUT = 3600

DiscardCallback: TypeAlias = Callable[[], None]
ContentsReader: TypeAlias = Callable[[], Awaitable[bytes]]
CacheFactory: TypeAlias = Callable[[pathlib.Path, Optional[DiscardCallback]], "FileCache"]


def bind_cache_to_remote_file(cache: "FileCache", file: RemoteFile) -> "RemoteFileWithCache":
    return RemoteFileWithCache.from_remote_file_and_cache(file, cache)


def create_file_cache(cache_path: pathlib.Path, discard_callback: Optional[DiscardCallback]) -> "FileCache":
    return FileCache(
        cache_path=cache_path,
        active=False,
        discard_callback=discard_callback,
        expire_after=_DEFAULT_CACHE_TIMEOUT,
    )


def create_default_file_cache_manager() -> "FileCacheManager":
    return FileCacheManager(cache_factory=create_file_cache)


@final
class FileCache(object):
    def __init__(
        self,
        *,
        cache_path: pathlib.Path,
        active: bool,
        discard_callback: Optional[DiscardCallback],
        expire_after: Optional[float],
    ) -> None:
        super().__init__()

        self._cache_path = cache_path
        self._active = active
        self._discard_callback = discard_callback
        self._expire_after = expire_after

        self._lock = asyncio.Lock()
        self._discarded = False
        self._finalizer: Optional[weakref.finalize]
        if self._discard_callback is not None:
            self._finalizer = weakref.finalize(self, self._discard_callback)
        else:
            self._finalizer = None
        self._expire_handle: Optional[asyncio.TimerHandle] = None
        if self._active:
            self.activate()

    @property
    def cache_path(self) -> pathlib.Path:
        return self._cache_path

    @property
    def active(self) -> bool:
        return self._active

    @property
    def discarded(self) -> bool:
        return self._discarded

    @property
    def alive(self) -> bool:
        return not self._discarded

    def __del__(self) -> None:
        self._on_discard()

    def __copy__(self) -> NoReturn:
        raise RuntimeError(f"{self.__class__.__name__} is not copyable.")

    def __deepcopy__(self, memo: Any) -> NoReturn:
        raise RuntimeError(f"{self.__class__.__name__} is not deepcopyable.")

    async def fetch_or_update_contents(self, contents_reader: ContentsReader) -> bytes:
        if self._discarded:
            raise CacheDiscardedError
        async with self._lock:
            if self._discarded:
                raise CacheDiscardedError
            if not self._active:
                contents = await self._update_contents(await contents_reader())
                self.activate()
            else:
                contents = await self._fetch_contents()
            return contents

    async def update_contents(self, contents_reader: ContentsReader) -> bytes:
        if self._discarded:
            raise CacheDiscardedError
        new_contents = await contents_reader()
        async with self._lock:
            if self._discarded:
                raise CacheDiscardedError
            self.deactivate()
            contents = await self._update_contents(new_contents)
            self.activate()
            return contents

    def activate(self) -> None:
        def _expire_callback(cache_ref: weakref.ReferenceType) -> None:
            cache = cache_ref()
            if cache is not None:
                cache._deactivate()

        if self._discarded:
            raise CacheDiscardedError
        self._cancel_expire_callback()
        # Should we inject the event loop from outside?
        loop = asyncio.get_running_loop()
        if self._expire_after is not None:
            self._expire_handle = loop.call_later(self._expire_after, _expire_callback, weakref.ref(self))
        self._active = True

    def deactivate(self) -> None:
        self._cancel_expire_callback()
        self._deactivate()

    async def discard(self) -> None:
        async with self._lock:
            if not self._discarded:
                self._on_discard()
                self._discarded = True

    async def _fetch_contents(self) -> bytes:
        return await anyio.Path(self.cache_path).read_bytes()

    async def _update_contents(self, new_contents: bytes) -> bytes:
        async with await anyio.open_file(self.cache_path, "wb") as f:
            await f.write(new_contents)
        return new_contents

    def _deactivate(self) -> None:
        self._active = False

    def _on_discard(self) -> None:
        self.deactivate()
        if self._discard_callback is not None:
            if self._finalizer is not None:
                self._finalizer.detach()
            self._discard_callback()

    def _cancel_expire_callback(self) -> None:
        if self._expire_handle is not None:
            self._expire_handle.cancel()
            self._expire_handle = None


@final
class FileCacheManager(object):
    def __init__(self, cache_factory: CacheFactory):
        super().__init__()
        self._cache_factory = cache_factory
        self._file_id_to_cache: weakref.WeakValueDictionary[str, FileCache] = weakref.WeakValueDictionary()
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    async def get_or_create_cache(
        self,
        file_id: str,
        cache_path: pathlib.Path,
        *,
        discard_callback: Optional[DiscardCallback] = None,
        init_cache_in_sync: Optional[bool] = None,
    ) -> Tuple[FileCache, bool]:
        if self._closed:
            raise RuntimeError("File cache manager is closed.")
        cache = None
        if self._has_cache(file_id):
            cache = self._get_cache(file_id)
        if cache is not None and cache.alive:
            return cache, False
        else:
            cache = self._create_cache(
                file_id,
                cache_path,
                discard_callback=discard_callback,
                init_cache_in_sync=init_cache_in_sync,
            )
            return cache, True

    async def get_cache(
        self,
        file_id: str,
    ) -> FileCache:
        if self._closed:
            raise RuntimeError("File cache manager is closed.")
        try:
            return self._get_cache(file_id)
        except KeyError as e:
            raise CacheNotFoundError from e

    async def remove_cache_if_exists(self, file_id: str) -> None:
        if self._closed:
            raise RuntimeError("File cache manager is closed.")
        if self._has_cache(file_id):
            cache = self._get_cache(file_id)
            await cache.discard()
            self._delete_cache(file_id)

    async def close(self) -> None:
        if not self._closed:
            for cache in self._file_id_to_cache.values():
                await cache.discard()
            self._clear_caches()
            self._closed = True

    def _create_cache(
        self,
        file_id: str,
        cache_path: pathlib.Path,
        *,
        discard_callback: Optional[DiscardCallback],
        init_cache_in_sync: Optional[bool],
    ) -> FileCache:
        cache = self._cache_factory(cache_path, discard_callback)
        if init_cache_in_sync is not None:
            if init_cache_in_sync:
                cache.activate()
            else:
                cache.deactivate()
        self._set_cache(file_id, cache)
        return cache

    def _has_cache(self, file_id: str) -> bool:
        return file_id in self._file_id_to_cache

    def _get_cache(self, file_id: str) -> FileCache:
        return self._file_id_to_cache[file_id]

    def _set_cache(self, file_id: str, cache: FileCache) -> None:
        self._file_id_to_cache[file_id] = cache

    def _delete_cache(self, file_id: str) -> None:
        del self._file_id_to_cache[file_id]

    def _clear_caches(self) -> None:
        self._file_id_to_cache.clear()


class RemoteFileWithCache(RemoteFile):
    def __init__(
        self,
        cache: FileCache,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._cache = cache

    @classmethod
    def from_remote_file_and_cache(cls, file: RemoteFile, cache: FileCache) -> Self:
        return cls(
            cache,
            id=file.id,
            filename=file.filename,
            byte_size=file.byte_size,
            created_at=file.created_at,
            purpose=file.purpose,
            metadata=file.metadata,
            client=file.client,
        )

    @property
    def cached(self) -> bool:
        return self._cache.alive and self._cache.active

    @property
    def cache_path(self) -> Optional[pathlib.Path]:
        return self._cache.cache_path if self._cache.alive else None

    async def read_contents(self) -> bytes:
        try:
            return await self._cache.fetch_or_update_contents(super().read_contents)
        except CacheDiscardedError:
            return await super().read_contents()

    async def delete(self) -> None:
        await super().delete()
        await self._cache.discard()

    async def update_cache(self) -> None:
        try:
            await self._cache.update_contents(super().read_contents)
        except CacheDiscardedError:
            logger.warning("Cache is no longer available.")


class CacheDiscardedError(Exception):
    pass


class CacheNotFoundError(Exception):
    pass
