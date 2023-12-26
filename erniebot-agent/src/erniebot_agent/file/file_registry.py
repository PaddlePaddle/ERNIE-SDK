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

from typing import Dict, Generic, List, Optional, TypeVar, final

from erniebot_agent.file.base import File

_T = TypeVar("_T", bound=File)


@final
class FileRegistry(Generic[_T]):
    def __init__(self) -> None:
        super().__init__()
        self._id_to_file: Dict[str, _T] = {}

    def register_file(self, file: _T, *, allow_overwrite: bool = False) -> None:
        file_id = file.id
        if file_id in self._id_to_file:
            if not allow_overwrite:
                raise ValueError(f"File with ID {repr(file_id)} is already registered.")
        self._id_to_file[file_id] = file

    def unregister_file(self, file: _T) -> None:
        file_id = file.id
        if file_id not in self._id_to_file:
            raise ValueError(f"File with ID {repr(file_id)} is not registered.")
        self._id_to_file.pop(file_id)

    def look_up_file(self, file_id: str) -> Optional[_T]:
        return self._id_to_file.get(file_id, None)

    def list_files(self) -> List[_T]:
        return list(self._id_to_file.values())
