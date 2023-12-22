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
import atexit
from typing import Any, List, Optional

from erniebot_agent.file.file_manager import FileManager
from erniebot_agent.file.remote_file import AIStudioFileClient
from erniebot_agent.utils import config_from_environ as C
from erniebot_agent.utils.mixins import Closeable

_global_file_manager: Optional[FileManager] = None
_objects_to_close: List[Closeable] = []


def get_global_file_manager() -> FileManager:
    global _global_file_manager
    if _global_file_manager is None:
        _global_file_manager = _create_default_file_manager(access_token=None, save_dir=None)
    return _global_file_manager


def configure_global_file_manager(
    access_token: Optional[str] = None, save_dir: Optional[str] = None, **opts: Any
) -> None:
    global _global_file_manager
    if _global_file_manager is not None:
        raise RuntimeError(
            "The global file manager can only be configured once before calling `get_global_file_manager`."
        )
    _global_file_manager = _create_default_file_manager(access_token=access_token, save_dir=save_dir, **opts)


def _create_default_file_manager(
    access_token: Optional[str], save_dir: Optional[str], **opts: Any
) -> FileManager:
    if access_token is None:
        access_token = C.get_global_access_token()
    if save_dir is None:
        save_dir = C.get_global_save_dir()
    if access_token is not None:
        remote_file_client = AIStudioFileClient(access_token=access_token)
    else:
        remote_file_client = None
    file_manager = FileManager(remote_file_client, save_dir, **opts)
    _objects_to_close.append(file_manager)
    return file_manager


def _close_objects():
    async def _close_objects_sequentially():
        for obj in _objects_to_close:
            await obj.close()

    if _objects_to_close:
        # Since async atexit is not officially supported by Python,
        # we start a new event loop to do the cleanup.
        asyncio.run(_close_objects_sequentially())
        _objects_to_close.clear()


# FIXME: The exit handler may not be called when using multiprocessing.
atexit.register(_close_objects)
