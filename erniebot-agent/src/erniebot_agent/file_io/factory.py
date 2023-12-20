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
from typing import List, Optional

from erniebot_agent.file_io.file_manager import FileManager
from erniebot_agent.file_io.remote_file import AIStudioFileClient
from erniebot_agent.utils.mixins import Closeable
from erniebot_agent.utils.temp_file import create_tracked_temp_dir

_objects_to_close: List[Closeable] = []


@functools.lru_cache(maxsize=None)
def get_global_file_manager(*, access_token: Optional[str]) -> FileManager:
    if access_token is None:
        file_manager = FileManager(save_dir=create_tracked_temp_dir())
    else:
        remote_file_client = AIStudioFileClient(access_token=access_token)
        _objects_to_close.append(remote_file_client)
        file_manager = FileManager(remote_file_client, save_dir=create_tracked_temp_dir())
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
