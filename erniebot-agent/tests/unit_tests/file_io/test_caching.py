import asyncio
import contextlib
import copy
import os
import pathlib
import tempfile
from unittest import mock

import pytest
from erniebot_agent.file_io.caching import (
    CacheDiscardedError,
    CacheNotFoundError,
    FileCache,
    FileCacheManager,
)

from tests.unit_tests.testing_utils.mocks.mock_remote_file_client_server import (
    FakeRemoteFileClient,
    FakeRemoteFileServer,
)


@contextlib.asynccontextmanager
async def create_file_cache_manager(cache_factory=None):
    if cache_factory is None:
        cache_factory = create_file_cache
    manager = FileCacheManager(cache_factory=cache_factory)

    yield manager

    await manager.close()


def create_file_cache(cache_path, active, discard_callback=None, expire_after=None):
    return FileCache(
        cache_path=cache_path,
        active=active,
        discard_callback=discard_callback,
        expire_after=expire_after,
    )


def repeat_bytes_in_coro_func(bytes_):
    async def _get_bytes():
        return bytes_

    return _get_bytes


@contextlib.contextmanager
def create_temporary_file():
    fd, path = tempfile.mkstemp()
    os.close(fd)
    try:
        yield pathlib.Path(path)
    finally:
        if os.path.exists(path):
            os.unlink(path)


class AwaitableContents(object):
    def __init__(self, contents):
        super().__init__()
        self.contents = contents

    def __await__(self):
        yield
        return self.contents


@pytest.mark.asyncio
async def test_file_cache_manager_crd():
    server = FakeRemoteFileServer()

    with server.start():
        client = FakeRemoteFileClient(server)

        with client.protocol.follow():
            with create_temporary_file() as file_path, create_temporary_file() as cache_path:
                with open(file_path, "wb") as f:
                    f.write(b"Simple is better than complex.")
                file = await client.upload_file(file_path, "assistants", {})

                async with create_file_cache_manager() as manager:
                    cache, created = await manager.get_or_create_cache(file.id, cache_path=cache_path)
                    assert cache.cache_path.samefile(cache_path)
                    assert created

                    retrieved_cache, created = await manager.get_or_create_cache(
                        file.id, cache_path=cache_path
                    )
                    assert retrieved_cache is cache
                    assert not created

                    retrieved_cache = await manager.get_cache(file.id)
                    assert retrieved_cache is cache

                    await manager.remove_cache_if_exists(file.id)
                    with pytest.raises(CacheNotFoundError):
                        await manager.get_cache(file.id)


@pytest.mark.asyncio
async def test_file_cache_manager_close():
    server = FakeRemoteFileServer()

    with server.start():
        client = FakeRemoteFileClient(server)

        with client.protocol.follow():
            with create_temporary_file() as file_path, create_temporary_file() as cache_path:
                with open(file_path, "wb") as f:
                    f.write(b"Simple is better than complex.")
                file = await client.upload_file(file_path, "assistants", {})

                async with create_file_cache_manager() as manager:
                    cache, _ = await manager.get_or_create_cache(file.id, cache_path=cache_path)

                    await manager.close()

                    assert manager.closed
                    assert cache.discarded


@pytest.mark.asyncio
async def test_file_cache_manager_after_closing():
    server = FakeRemoteFileServer()

    with server.start():
        client = FakeRemoteFileClient(server)

        with client.protocol.follow():
            with create_temporary_file() as file_path, create_temporary_file() as cache_path:
                with open(file_path, "wb") as f:
                    f.write(b"Flat is better than nested.")
                file = await client.upload_file(file_path, "assistants", {})

                async with create_file_cache_manager() as manager:
                    await manager.close()

                    with pytest.raises(RuntimeError):
                        await manager.get_or_create_cache(file.id, cache_path=cache_path)

                    with pytest.raises(RuntimeError):
                        await manager.get_cache(file.id)

                    with pytest.raises(RuntimeError):
                        await manager.remove_cache_if_exists(file.id)


@pytest.mark.asyncio
async def test_file_cache_manager_auto_remove_unreachable_cache():
    server = FakeRemoteFileServer()

    with server.start():
        client = FakeRemoteFileClient(server)

        with client.protocol.follow():
            with create_temporary_file() as file_path, create_temporary_file() as cache_path:
                with open(file_path, "wb") as f:
                    f.write(b"Flat is better than nested.")
                file = await client.upload_file(file_path, "assistants", {})

                async with create_file_cache_manager() as manager:
                    cache, _ = await manager.get_or_create_cache(file.id, cache_path=cache_path)

                    del cache

                    with pytest.raises(CacheNotFoundError):
                        await manager.get_cache(file.id)


@pytest.mark.parametrize("active", [False, True])
@pytest.mark.asyncio
async def test_file_cache_init_active(active):
    with create_temporary_file() as cache_path:
        cache = create_file_cache(
            cache_path=cache_path,
            active=active,
        )

        if active:
            assert cache.active
        else:
            assert not cache.active


@pytest.mark.asyncio
async def test_file_cache_timeout():
    expire_after = 0.1

    with create_temporary_file() as cache_path:
        cache = create_file_cache(
            cache_path=cache_path,
            active=True,
            expire_after=expire_after,
        )
        await asyncio.sleep(expire_after * 1.5)

        assert not cache.active


@pytest.mark.asyncio
async def test_file_cache_fetch_or_update_contents():
    with create_temporary_file() as cache_path:
        cache = create_file_cache(
            cache_path=cache_path,
            active=False,
        )

        contents1 = b"Special cases aren't special enough to break the rules."
        result = await cache.fetch_or_update_contents(repeat_bytes_in_coro_func(contents1))
        assert result == contents1
        assert cache.cache_path.read_bytes() == contents1

        contents2 = b"Although practicality beats purity."
        result = await cache.fetch_or_update_contents(repeat_bytes_in_coro_func(contents2))
        assert result == contents1
        assert cache.cache_path.read_bytes() == contents1


@pytest.mark.asyncio
async def test_file_cache_update_contents():
    with create_temporary_file() as cache_path:
        cache = create_file_cache(
            cache_path=cache_path,
            active=False,
        )

        contents1 = b"Errors should never pass silently."
        result = await cache.update_contents(repeat_bytes_in_coro_func(contents1))
        assert result == contents1
        assert cache.cache_path.read_bytes() == contents1

        contents2 = b"Unless explicitly silenced."
        result = await cache.update_contents(repeat_bytes_in_coro_func(contents2))
        assert result == contents2
        assert cache.cache_path.read_bytes() == contents2


@pytest.mark.asyncio
async def test_file_cache_activate_deactivate():
    with create_temporary_file() as cache_path:
        cache = create_file_cache(
            cache_path=cache_path,
            active=False,
        )

        cache.activate()
        assert cache.active

        cache.deactivate()
        assert not cache.active


@pytest.mark.asyncio
async def test_file_cache_discard():
    with create_temporary_file() as cache_path:
        cache = create_file_cache(
            cache_path=cache_path,
            active=True,
        )

        await cache.discard()

        assert cache.discarded
        assert not cache.alive
        assert not cache.active


@pytest.mark.asyncio
async def test_file_cache_fetch_or_update_contents_called_concurrently():
    contents = b"Explicit is better than implicit."
    num_coros = 4

    mock_ = mock.Mock(return_value=AwaitableContents(contents))

    with create_temporary_file() as cache_path:
        cache = create_file_cache(
            cache_path=cache_path,
            active=False,
        )

        results = await asyncio.gather(*[cache.fetch_or_update_contents(mock_) for _ in range(num_coros)])

        for result in results:
            assert result == contents
        assert mock_.call_count == 1


@pytest.mark.asyncio
async def test_file_cache_update_contents_called_concurrently():
    contents = b"Readability counts."
    num_coros = 4

    mock_ = mock.Mock(return_value=AwaitableContents(contents))

    with create_temporary_file() as cache_path:
        cache = create_file_cache(
            cache_path=cache_path,
            active=False,
        )

        results = await asyncio.gather(*[cache.update_contents(mock_) for _ in range(num_coros)])

        for result in results:
            assert result == contents
        assert mock_.call_count == num_coros


@pytest.mark.asyncio
async def test_file_cache_discard_called_concurrently():
    num_coros = 4

    mock_ = mock.Mock()

    with create_temporary_file() as cache_path:
        cache = create_file_cache(
            cache_path=cache_path,
            active=False,
            discard_callback=mock_,
        )

        await asyncio.gather(*[cache.discard() for _ in range(num_coros)])

        assert cache.discarded
        assert mock_.call_count == 1


@pytest.mark.asyncio
async def test_file_cache_discard_callback():
    mock_ = mock.Mock()

    with create_temporary_file() as cache_path:
        cache = create_file_cache(
            cache_path=cache_path,
            active=False,
            discard_callback=mock_,
        )

        await cache.discard()
        mock_.assert_called_once_with()


@pytest.mark.asyncio
async def test_file_cache_discard_on_destruction():
    mock_ = mock.Mock()

    with create_temporary_file() as cache_path:
        cache = create_file_cache(
            cache_path=cache_path,
            active=False,
            discard_callback=mock_,
        )

        del cache
        mock_.assert_called_once_with()


@pytest.mark.asyncio
async def test_file_cache_after_discarding():
    with create_temporary_file() as cache_path:
        cache = create_file_cache(
            cache_path=cache_path,
            active=True,
        )

        await cache.discard()

        with pytest.raises(CacheDiscardedError):
            await cache.fetch_or_update_contents(
                repeat_bytes_in_coro_func(b"If the implementation is hard to explain, it's a bad idea.")
            )

        with pytest.raises(CacheDiscardedError):
            await cache.update_contents(
                repeat_bytes_in_coro_func(
                    b"If the implementation is easy to explain, it may be a good idea."
                )
            )

        with pytest.raises(CacheDiscardedError):
            cache.activate()


@pytest.mark.asyncio
async def test_file_cache_copy():
    with create_temporary_file() as cache_path:
        cache = create_file_cache(
            cache_path=cache_path,
            active=False,
        )

        with pytest.raises(RuntimeError):
            copy.copy(cache)

        with pytest.raises(RuntimeError):
            copy.deepcopy(cache)
