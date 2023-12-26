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
from typing import Any, NoReturn, Optional, final

import asyncio_atexit  # type: ignore

from erniebot_agent.file.file_manager import FileManager
from erniebot_agent.file.remote_file import AIStudioFileClient
from erniebot_agent.utils import config_from_environ as C
from erniebot_agent.utils.misc import SingletonMeta


@final
class GlobalFileManagerHandler(metaclass=SingletonMeta):
    _file_manager: Optional[FileManager]

    def __init__(self) -> None:
        super().__init__()
        self._lock = asyncio.Lock()
        self._file_manager = None

    async def get(self) -> FileManager:
        async with self._lock:
            if self._file_manager is None:
                self._file_manager = await self._create_default_file_manager(
                    access_token=None,
                    save_dir=None,
                    enable_remote_file=False,
                )
            return self._file_manager

    async def configure(
        self,
        *,
        access_token: Optional[str] = None,
        save_dir: Optional[str] = None,
        enable_remote_file: bool = False,
        **opts: Any,
    ) -> None:
        async with self._lock:
            if self._file_manager is not None:
                self._raise_file_manager_already_set_error()
            self._file_manager = await self._create_default_file_manager(
                access_token=access_token,
                save_dir=save_dir,
                enable_remote_file=enable_remote_file,
                **opts,
            )

    async def set(self, file_manager: FileManager) -> None:
        async with self._lock:
            if self._file_manager is not None:
                self._raise_file_manager_already_set_error()
            self._file_manager = file_manager

    async def _create_default_file_manager(
        self,
        access_token: Optional[str],
        save_dir: Optional[str],
        enable_remote_file: bool,
        **opts: Any,
    ) -> FileManager:
        async def _close_file_manager():
            await file_manager.close()

        if access_token is None:
            access_token = C.get_global_access_token()
        if save_dir is None:
            save_dir = C.get_global_save_dir()
        remote_file_client = None
        if enable_remote_file:
            if access_token is None:
                raise RuntimeError("An access token must be provided to enable remote file management.")
            remote_file_client = AIStudioFileClient(access_token=access_token)
        file_manager = FileManager(remote_file_client, save_dir, **opts)
        asyncio_atexit.register(_close_file_manager)
        return file_manager

    def _raise_file_manager_already_set_error(self) -> NoReturn:
        raise RuntimeError(
            "The global file manager can only be set once."
            " The setup can be done explicitly by calling"
            " `GlobalFileManagerHandler.configure` or `GlobalFileManagerHandler.set`,"
            " or implicitly by calling `GlobalFileManagerHandler.get`."
        )
