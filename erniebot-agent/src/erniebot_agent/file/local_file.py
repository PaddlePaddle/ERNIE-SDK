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

import pathlib
import uuid
from typing import Any, Dict

import anyio

from erniebot_agent.file import protocol
from erniebot_agent.file.base import BaseFile


def create_local_file_from_path(
    file_path: pathlib.Path,
    file_purpose: protocol.FilePurpose,
    file_metadata: Dict[str, Any],
) -> "LocalFile":
    """
    Create a LocalFile object from a local file path.

    Args:
        file_path (pathlib.Path): The path to the local file.
        file_purpose (protocol.FilePurpose): The purpose or use case of the file,
                                             including "assistants" and "assistants_output".
        file_metadata (Dict[str, Any]): Additional metadata associated with the file.

    Returns:
        LocalFile: The created LocalFile object.

    Raises:
        FileNotFoundError: If the specified file does not exist.

    """
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")
    file_id = _generate_local_file_id()
    filename = file_path.name
    byte_size = file_path.stat().st_size
    created_at = protocol.get_timestamp()
    file = LocalFile(
        id=file_id,
        filename=filename,
        byte_size=byte_size,
        created_at=created_at,
        purpose=file_purpose,
        metadata=file_metadata,
        path=file_path,
    )
    return file


class LocalFile(BaseFile):
    """
    Represents a local file.

    Attributes:
        id (str): Unique identifier for the file.
        filename (str): File name.
        byte_size (int): Size of the file in bytes.
        created_at (str): Timestamp indicating the file creation time.
        purpose (str): Purpose or use case of the file,
                       including "assistants" and "assistants_output".
        metadata (Dict[str, Any]): Additional metadata associated with the file.
        path (pathlib.Path): The path to the local file.

    Methods:
        read_contents: Asynchronously read the contents of the local file.
        write_contents_to: Asynchronously write the file contents to a local path.
        get_file_repr: Return a string representation for use in specific contexts.

    """

    def __init__(
        self,
        *,
        id: str,
        filename: str,
        byte_size: int,
        created_at: str,
        purpose: protocol.FilePurpose,
        metadata: Dict[str, Any],
        path: pathlib.Path,
        validate_file_id: bool = True,
    ) -> None:
        """
        Initialize a LocalFile object.

        Args:
            id (str): The unique identifier for the file.
            filename (str): The name of the file.
            byte_size (int): The size of the file in bytes.
            created_at (str): The timestamp indicating the file creation time.
            purpose (protocol.FilePurpose): The purpose or use case of the file.
            metadata (Dict[str, Any]): Additional metadata associated with the file.
            path (pathlib.Path): The path to the local file.
            validate_file_id (bool): Flag to validate the file ID. Default is True.

        Raises:
            ValueError: If the file ID is invalid.

        """
        if validate_file_id:
            if not protocol.is_local_file_id(id):
                raise ValueError(f"Invalid file ID: {id}")
        if not protocol.is_valid_file_purpose(purpose):
            raise ValueError(f"Invalid file purpose: {purpose}")
        super().__init__(
            id=id,
            filename=filename,
            byte_size=byte_size,
            created_at=created_at,
            purpose=purpose,
            metadata=metadata,
        )
        self.path = path

    async def read_contents(self) -> bytes:
        """Asynchronously read the contents of the local file."""
        return await anyio.Path(self.path).read_bytes()

    def _get_attrs_str(self) -> str:
        attrs_str = super()._get_attrs_str()
        attrs_str += f", path: {repr(self.path)}"
        return attrs_str


def _generate_local_file_id():
    return protocol.create_local_file_id_from_uuid(str(uuid.uuid1()))
