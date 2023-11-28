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

import threading
from typing import Dict, List, Optional

from erniebot_agent.file_io.base import File
from erniebot_agent.utils.misc import Singleton


class FileRegistry(metaclass=Singleton):
    def __init__(self) -> None:
        super().__init__()
        self._id_to_file: Dict[str, File] = {}
        self._lock = threading.Lock()

    def register_file(self, file: File) -> None:
        file_id = file.id
        with self._lock:
            # Re-registering an existing file is allowed.
            # We simply update the registry.
            self._id_to_file[file_id] = file

    def unregister_file(self, file: File) -> None:
        file_id = file.id
        with self._lock:
            if file_id not in self._id_to_file:
                raise RuntimeError(f"ID {repr(file_id)} is not registered.")
            self._id_to_file.pop(file_id)

    def lookup_file(self, file_id: str) -> Optional[File]:
        with self._lock:
            return self._id_to_file.get(file_id, None)

    def list_files(self) -> List[File]:
        with self._lock:
            return list(self._id_to_file.values())


_file_registry = FileRegistry()


def get_file_registry() -> FileRegistry:
    return _file_registry
