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
from typing import Optional

from erniebot_agent.file_io.file_manager import FileManager
from erniebot_agent.file_io.remote_file import AIStudioFileClient


@functools.lru_cache(maxsize=None)
def get_file_manager(access_token: Optional[str] = None) -> FileManager:
    if access_token is None:
        # TODO: Use a default global access token.
        return FileManager()
    else:
        remote_file_client = AIStudioFileClient(access_token=access_token)
        return FileManager(remote_file_client)
