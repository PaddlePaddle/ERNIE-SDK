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

"""
File module is used to manage the file system by a global file_manager.
Including `local file` and `remote file`.

A few notes about the current state of this submodule:

- If you do not set environment variable `AISTUDIO_ACCESS_TOKEN`, it will be under default setting.

- Method `configure_global_file_manager` can only be called once at the beginning.

- When you want to get a file manger, you can use method `get_global_file_manager`.

- If you want to get the content of `File` object, you can use `read_contents`
  and use `write_contents_to` create the file to location you want.
"""

from erniebot_agent.file.global_file_manager_handler import GlobalFileManagerHandler
from erniebot_agent.file.base import File
from erniebot_agent.file.file_manager import FileManager
from erniebot_agent.file.local_file import LocalFile
from erniebot_agent.file.remote_file import RemoteFile