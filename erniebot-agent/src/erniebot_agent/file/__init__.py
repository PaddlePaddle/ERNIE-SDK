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
File module is used to manage the file system by a global file manager: `GlobalFileManagerHandler`.

We use it to manage `File` obeject, including `LocalFile` and `RemoteFile`.

A few notes about this submodule:

- If you do not set environment variable `AISTUDIO_ACCESS_TOKEN`, it will be under default setting.

- Method `GlobalFileManagerHandler().configure()` can only be called **once** at the beginning.

- When you want to get a file manger, you can use method `GlobalFileManagerHandler().get()`.

- The lifecycle of the `FileManager` class is synchronized with the event loop.

- `FileManager` class is Noncopyable.

- If you want to get the content of `File` object, you can use `read_contents`
  and use `write_contents_to` create the file to location you want.

- We do **not** recommend you to create `File` object yourself.

Examples:
    >>> from erniebot_agent.file import GlobalFileManagerHandler
    >>> async def demo_function():
    >>>     file_manager = await GlobalFileManagerHandler().get()
    >>>     local_file = await file_manager.create_file_from_path(file_path='your_path', file_type='local')

    >>>     file = file_manager.look_up_file_by_id(file_id='your_file_id')
    >>>     file_content = await file.read_contents()
"""

from .global_file_manager_handler import GlobalFileManagerHandler
from .remote_file import AIStudioFileClient
