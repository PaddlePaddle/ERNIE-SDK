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

from erniebot_agent.file.base import BaseFile

_T = TypeVar("_T", bound=BaseFile)


@final
class FileRegistry(Generic[_T]):
    """
    Singleton class for managing file registration.


    Methods:
        register_file: Register a file in the registry.
        unregister_file: Unregister a file from the registry.
        look_up_file: Look up a file by its ID in the registry.
        list_files: Get a list of all registered files.

    """

    def __init__(self) -> None:
        super().__init__()
        self._id_to_file: Dict[str, _T] = {}

    def register_file(self, file: _T, *, allow_overwrite: bool = False) -> None:
        """
        Register a file in the registry.

        Args:
            file (File): The file object to register.
            allow_overwrite (bool): Allow overwriting if a file with the same ID is already registered.

        Returns:
            None

        Raises:
            RuntimeError: If the file ID is already registered and allow_overwrite is False.

        """
        file_id = file.id
        if file_id in self._id_to_file:
            if not allow_overwrite:
                raise ValueError(f"File with ID {repr(file_id)} is already registered.")
        self._id_to_file[file_id] = file

    def unregister_file(self, file: _T) -> None:
        """
        Unregister a file from the registry.

        Args:
            file (File): The file object to unregister.

        Returns:
            None

        Raises:
            RuntimeError: If the file ID is not registered.

        """
        file_id = file.id
        if file_id not in self._id_to_file:
            raise ValueError(f"File with ID {repr(file_id)} is not registered.")
        self._id_to_file.pop(file_id)

    def look_up_file(self, file_id: str) -> Optional[_T]:
        """
        Look up a file by its ID in the registry.

        Args:
            file_id (str): The ID of the file to look up.

        Returns:
            Optional[File]: The File object if found, or None if not found.

        """
        return self._id_to_file.get(file_id, None)

    def list_files(self) -> List[_T]:
        """
        Get a list of all registered files.

        Returns:
            List[File]: The list of registered File objects.

        """
        return list(self._id_to_file.values())
