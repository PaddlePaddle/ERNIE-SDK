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
import functools
from typing import Optional

from erniebot_agent.file_io.file_manager import FileManager
from erniebot_agent.file_io.file_registry import get_global_file_registry
from erniebot_agent.file_io.remote_file import AIStudioFileClient
from erniebot_agent.utils.temp_file import create_tracked_temp_dir


@functools.lru_cache(maxsize=None)
def get_global_file_manager(access_token: Optional[str] = None) -> FileManager:
    global_file_registry = get_global_file_registry()
    if access_token is None:
        # TODO: Use a default global access token.
        file_manager = FileManager(global_file_registry, save_dir=create_tracked_temp_dir())
    else:
        remote_file_client = AIStudioFileClient(access_token=access_token)
        file_manager = FileManager(
            global_file_registry, remote_file_client, save_dir=create_tracked_temp_dir()
        )
    atexit.register(_close_file_manager, file_manager)
    return file_manager


def _close_file_manager(file_manager: FileManager):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(file_manager.close())
    else:
        loop.create_task(file_manager.close())
