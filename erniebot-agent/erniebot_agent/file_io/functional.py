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
from typing import Literal, Optional, Union, overload

from erniebot_agent.file_io.base import File
from erniebot_agent.file_io.file_manager import FileManager, FilePath
from erniebot_agent.file_io.local_file import LocalFile
from erniebot_agent.file_io.protocol import FilePurpose
from erniebot_agent.file_io.remote_file import AIStudioFileClient, RemoteFile


@functools.lru_cache(maxsize=None)
def get_file_manager(access_token: Optional[str] = None) -> FileManager:
    if access_token is None:
        return FileManager()
    else:
        remote_file_client = AIStudioFileClient(access_token=access_token)
        return FileManager(remote_file_client)


@overload
async def create_file_from_path(
    file_path: FilePath,
    *,
    file_purpose: FilePurpose,
    file_meta: Optional[str] = ...,
    file_type: Literal["local"] = ...,
    access_token: Optional[str] = ...,
) -> LocalFile:
    ...


@overload
async def create_file_from_path(
    file_path: FilePath,
    *,
    file_purpose: FilePurpose,
    file_meta: Optional[str] = ...,
    file_type: Literal["remote"],
    access_token: str,
) -> RemoteFile:
    ...


async def create_file_from_path(
    file_path: FilePath,
    *,
    file_purpose: FilePurpose,
    file_meta: Optional[str] = None,
    file_type: Literal["local", "remote"] = "local",
    access_token: Optional[str] = None,
) -> Union[LocalFile, RemoteFile]:
    file_manager = get_file_manager(access_token=access_token)
    return await file_manager.create_file_from_path(
        file_path,
        file_purpose=file_purpose,
        file_meta=file_meta,
        file_type=file_type,
    )


@overload
async def create_file_from_bytes(
    file_contents: bytes,
    filename: str,
    *,
    file_purpose: FilePurpose,
    file_meta: Optional[str] = ...,
    file_type: Literal["local"] = ...,
    access_token: Optional[str] = ...,
) -> LocalFile:
    ...


@overload
async def create_file_from_bytes(
    file_contents: bytes,
    filename: str,
    *,
    file_purpose: FilePurpose,
    file_meta: Optional[str] = ...,
    file_type: Literal["remote"],
    access_token: str,
) -> RemoteFile:
    ...


async def create_file_from_bytes(
    file_contents: bytes,
    filename: str,
    *,
    file_purpose: FilePurpose,
    file_meta: Optional[str] = None,
    file_type: Literal["local", "remote"] = "local",
    access_token: Optional[str] = None,
) -> Union[LocalFile, RemoteFile]:
    file_manager = get_file_manager(access_token=access_token)
    return await file_manager.create_file_from_bytes(
        file_contents,
        filename,
        file_purpose=file_purpose,
        file_meta=file_meta,
        file_type=file_type,
    )


async def retrieve_remote_file_by_id(file_id: str, access_token: str) -> RemoteFile:
    file_manager = get_file_manager(access_token=access_token)
    return await file_manager.retrieve_remote_file_by_id(file_id)


def look_up_file_by_id(file_id: str, access_token: Optional[str] = None) -> Optional[File]:
    file_manager = get_file_manager(access_token=access_token)
    return file_manager.look_up_file_by_id(file_id)
