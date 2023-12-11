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

from typing import Dict, List, Optional, final

from erniebot_agent.file_io.base import File
from erniebot_agent.utils.misc import Singleton


class BaseFileRegistry(object):
    def register_file(self, file: File, *, allow_overwrite: bool = False) -> None:
        raise NotImplementedError

    def unregister_file(self, file: File) -> None:
        raise NotImplementedError

    def look_up_file(self, file_id: str) -> Optional[File]:
        raise NotImplementedError

    def list_files(self) -> List[File]:
        raise NotImplementedError


class FileRegistry(BaseFileRegistry):
    def __init__(self) -> None:
        super().__init__()
        self._id_to_file: Dict[str, File] = {}

    def register_file(self, file: File, *, allow_overwrite: bool = False) -> None:
        file_id = file.id
        if not allow_overwrite and file_id in self._id_to_file:
            raise RuntimeError(f"ID {repr(file_id)} is already registered.")
        self._id_to_file[file_id] = file

    def unregister_file(self, file: File) -> None:
        file_id = file.id
        if file_id not in self._id_to_file:
            raise RuntimeError(f"ID {repr(file_id)} is not registered.")
        self._id_to_file.pop(file_id)

    def look_up_file(self, file_id: str) -> Optional[File]:
        return self._id_to_file.get(file_id, None)

    def list_files(self) -> List[File]:
        return list(self._id_to_file.values())


@final
class GlobalFileRegistry(FileRegistry, metaclass=Singleton):
    pass


def get_global_file_registry() -> GlobalFileRegistry:
    return GlobalFileRegistry()
