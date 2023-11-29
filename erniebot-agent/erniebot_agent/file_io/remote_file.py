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

import abc
import pathlib
from typing import List

from erniebot_agent.file_io.base import File
from erniebot_agent.file_io.protocol import is_remote_file_id


class RemoteFile(File):
    def __init__(self, id: str, filename: str, created_at: int, client: "RemoteFileClient") -> None:
        if not is_remote_file_id(id):
            raise ValueError("Invalid file ID: {id}")
        super().__init__(id=id, filename=filename, created_at=created_at)
        self._client = client

    async def read_contents(self) -> bytes:
        file_contents = await self._client.retrieve_file_contents(self.id)
        return file_contents

    async def delete(self) -> None:
        await self._client.delete_file(self.id)


class RemoteFileClient(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    async def upload_file(self, file_path: pathlib.Path) -> RemoteFile:
        raise NotImplementedError

    @abc.abstractmethod
    async def retrieve_file(self, file_id: str) -> RemoteFile:
        raise NotImplementedError

    @abc.abstractmethod
    async def retrieve_file_contents(self, file_id: str) -> bytes:
        raise NotImplementedError

    @abc.abstractmethod
    async def list_files(self) -> List[RemoteFile]:
        raise NotImplementedError

    @abc.abstractmethod
    async def delete_file(self, file_id: str) -> None:
        raise NotImplementedError


import asyncio
import functools
import pathlib
import time
import uuid
from typing import ClassVar, Dict

import anyio
from baidubce.auth.bce_credentials import BceCredentials
from baidubce.bce_client_configuration import BceClientConfiguration
from baidubce.services.bos.bos_client import BosClient
from erniebot_agent.file_io.protocol import build_remote_file_id_from_uuid


class BOSFileClient(RemoteFileClient):
    _ENDPOINT: ClassVar[str] = "bj.bcebos.com"

    def __init__(self, ak: str, sk: str, bucket_name: str, prefix: str) -> None:
        super().__init__()
        self.bucket_name = bucket_name
        self.prefix = prefix
        config = BceClientConfiguration(credentials=BceCredentials(ak, sk), endpoint=self._ENDPOINT)
        self._bos_client = BosClient(config=config)

    async def upload_file(self, file_path: pathlib.Path) -> RemoteFile:
        file_id = self._generate_file_id()
        filename = file_path.name
        created_at = int(time.time())
        user_metadata: Dict[str, str] = {"id": file_id, "filename": filename, "created_at": str(created_at)}
        async with await anyio.open_file(file_path, mode="rb") as f:
            data = await f.read()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            functools.partial(
                self._bos_client.put_object_from_string,
                bucket=self.bucket_name,
                key=self._get_key(file_id),
                data=data,
                user_metadata=user_metadata,
            ),
        )
        return RemoteFile(
            id=file_id,
            filename=filename,
            created_at=created_at,
            client=self,
        )

    async def retrieve_file(self, file_id: str) -> RemoteFile:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            functools.partial(
                self._bos_client.get_object_meta_data, self.bucket_name, self._get_key(file_id)
            ),
        )
        user_metadata = {
            "id": response.metadata.bce_meta_id,
            "filename": response.metadata.bce_meta_filename,
            "created_at": int(response.metadata.bce_meta_created_at),
        }
        if file_id != user_metadata["id"]:
            raise RuntimeError("`file_id` is not the same as the one in metadata.")

        return RemoteFile(
            id=user_metadata["id"],
            filename=user_metadata["filename"],
            created_at=user_metadata["created_at"],
            client=self,
        )

    async def retrieve_file_contents(self, file_id: str) -> bytes:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            functools.partial(
                self._bos_client.get_object_as_string, self.bucket_name, self._get_key(file_id)
            ),
        )
        return result

    async def list_files(self) -> List[RemoteFile]:
        raise RuntimeError(f"`{self.__class__.__name__}.list_files` is not supported.")

    async def delete_file(self, file_id: str) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, functools.partial(self._bos_client.delete_object, self.bucket_name, self._get_key(file_id))
        )

    def _get_key(self, file_id: str) -> str:
        return self.prefix + file_id

    @staticmethod
    def _generate_file_id() -> str:
        return build_remote_file_id_from_uuid(str(uuid.uuid1()))
