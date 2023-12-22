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
from typing import Any, Optional, final

import asyncio_atexit  # type: ignore

from erniebot_agent.file.file_manager import FileManager
from erniebot_agent.file.remote_file import AIStudioFileClient
from erniebot_agent.utils import config_from_environ as C
from erniebot_agent.utils.misc import Singleton


@final
class GlobalFileManager(metaclass=Singleton):
    _file_manager: Optional[FileManager]

    def __init__(self) -> None:
        super().__init__()
        self._lock = asyncio.Lock()
        self._file_manager = None

    async def get(self) -> FileManager:
        async with self._lock:
            if self._file_manager is None:
                self._file_manager = await self._create_default_file_manager(
                    access_token=None, save_dir=None
                )
        return self._file_manager

    async def configure(
        self,
        access_token: Optional[str] = None,
        save_dir: Optional[str] = None,
        **opts: Any,
    ) -> None:
        async with self._lock:
            if self._file_manager is not None:
                raise RuntimeError(
                    "`GlobalFileManager.configure` can only be called once"
                    " and must be called before calling `GlobalFileManager.get`."
                )
            self._file_manager = await self._create_default_file_manager(
                access_token=access_token, save_dir=save_dir, **opts
            )

    async def _create_default_file_manager(
        self,
        access_token: Optional[str],
        save_dir: Optional[str],
        **opts: Any,
    ) -> FileManager:
        async def _close_file_manager():
            await file_manager.close()

        if access_token is None:
            access_token = C.get_global_access_token()
        if save_dir is None:
            save_dir = C.get_global_save_dir()
        if access_token is not None:
            remote_file_client = AIStudioFileClient(access_token=access_token)
        else:
            remote_file_client = None
        file_manager = FileManager(remote_file_client, save_dir, **opts)
        asyncio_atexit.register(_close_file_manager)
        return file_manager
