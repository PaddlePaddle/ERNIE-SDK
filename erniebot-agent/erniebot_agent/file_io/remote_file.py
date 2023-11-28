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

import os
from typing import List, Union

from erniebot_agent.file_io.base import File
from erniebot_agent.file_io.protocol import is_remote_file_id
from erniebot_agent.file_io.remote_file_clients.base import RemoteFileClient
from erniebot_agent.file_io.remote_file_clients.schema import FileInfo


class RemoteFile(File):
    def __init__(self, id: str, filename: str, created_at: int, client: RemoteFileClient) -> None:
        if not is_remote_file_id(id):
            raise ValueError("Invalid file ID: {id}")
        super().__init__(id=id, filename=filename, created_at=created_at)
        self._client = client

    async def read_content(self) -> bytes:
        file_content = await self._client.retrieve_file_content(self.id)
        return file_content.content

    async def delete(self) -> None:
        await self._client.delete_file(self.id)


async def create_remote_file_from_path(
    file_path: Union[str, os.PathLike], client: RemoteFileClient
) -> RemoteFile:
    file_info = await client.upload_file(open(file_path, "rb"))
    return _build_remote_file_from_file_info(file_info, client)


async def retrieve_remote_file_by_id(file_id: str, client: RemoteFileClient) -> RemoteFile:
    file_info = await client.retrieve_file(file_id)
    return _build_remote_file_from_file_info(file_info, client)


async def list_remote_files(client: RemoteFileClient) -> List[RemoteFile]:
    file_info_list = await client.list_files()
    return [_build_remote_file_from_file_info(file_info, client) for file_info in file_info_list]


def _build_remote_file_from_file_info(file_info: FileInfo, client: RemoteFileClient) -> RemoteFile:
    return RemoteFile(
        id=file_info.id,
        filename=file_info.filename,
        created_at=file_info.created_at,
        client=client,
    )
