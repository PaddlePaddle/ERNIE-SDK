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
import inspect
import json
import os
import pathlib
from typing import Any, Dict, Final, List, Optional

import aiohttp

from erniebot_agent.file import protocol
from erniebot_agent.file.base import BaseFile
from erniebot_agent.utils.exceptions import FileError
from erniebot_agent.utils.mixins import Closeable


class RemoteFile(BaseFile):
    """
    Represents a remote file.

    Attributes:
        id (str): Unique identifier for the file.
        filename (str): File name.
        byte_size (int): Size of the file in bytes.
        created_at (str): Timestamp indicating the file creation time.
        purpose (str): Purpose or use case of the file,
                       including "assistants" and "assistants_output".
        metadata (Dict[str, Any]): Additional metadata associated with the file.
        client (RemoteFileClient): The client of remote file.

    Methods:
        read_contents: Asynchronously read the contents of the local file.
        write_contents_to: Asynchronously write the file contents to a local path.
        get_file_repr: Return a string representation for use in specific contexts.
        delete: Asynchronously delete the file from client.
        create_temporary_url: Asynchronously create a temporary URL for the file.

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
        client: "RemoteFileClient",
        validate_file_id: bool = True,
    ) -> None:
        if validate_file_id:
            if not protocol.is_remote_file_id(id):
                raise FileError(f"Invalid file ID: {id}")
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
        self._client = client

    @property
    def client(self) -> "RemoteFileClient":
        return self._client

    async def read_contents(self) -> bytes:
        file_contents = await self._client.retrieve_file_contents(self.id)
        return file_contents

    async def delete(self) -> None:
        await self._client.delete_file(self.id)

    async def create_temporary_url(self, expire_after: float = 600) -> str:
        """To create a temporary valid URL for the file."""
        return await self._client.create_temporary_url(self.id, expire_after)

    def get_file_repr_with_url(self, url: str) -> str:
        return f"{self.get_file_repr()}<url>{url}</url>"


class RemoteFileClient(Closeable, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    async def upload_file(
        self, file_path: pathlib.Path, file_purpose: protocol.FilePurpose, file_metadata: Dict[str, Any]
    ) -> RemoteFile:
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

    @abc.abstractmethod
    async def create_temporary_url(self, file_id: str, expire_after: float) -> str:
        raise NotImplementedError


class AIStudioFileClient(RemoteFileClient):
    """
    Recommended remote file client: AI Studio.

    Methods:
        upload_file: Upload a file to AI Studio client.
        retrieve_file: Retrieve information about a file from AI Studio.
        retrieve_file_contents: Retrieve the contents of a file from AI Studio.
        list_files: List files available in AI Studio.
        delete_file: Delete a file in AI Studio client(#TODO: not supported now).
        create_temporary_url: Create a temporary URL for accessing a file in AI Studio.
        close: Close the AIStudioFileClient.

    """

    _BASE_URL: Final[str] = "https://sandbox-aistudio.baidu.com"
    _UPLOAD_ENDPOINT: Final[str] = "/llm/lmapp/files"
    _RETRIEVE_ENDPOINT: Final[str] = "/llm/lmapp/files/{file_id}"
    _RETRIEVE_CONTENTS_ENDPOINT: Final[str] = "/llm/lmapp/files/{file_id}/content"
    _LIST_ENDPOINT: Final[str] = "/llm/lmapp/files"

    def __init__(
        self, access_token: str, *, aiohttp_session: Optional[aiohttp.ClientSession] = None
    ) -> None:
        """
        Initialize the AIStudioFileClient.

        Args:
            access_token (str): The access token for AI Studio.
            aiohttp_session (Optional[aiohttp.ClientSession]): A custom aiohttp session (default is None).

        """
        super().__init__()
        self._access_token = access_token
        if aiohttp_session is None:
            aiohttp_session = self._create_aiohttp_session()
        self._session = aiohttp_session
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    async def upload_file(
        self, file_path: pathlib.Path, file_purpose: protocol.FilePurpose, file_metadata: Dict[str, Any]
    ) -> RemoteFile:
        """Upload a file to AI Studio client."""
        self.ensure_not_closed()
        url = self._get_url(self._UPLOAD_ENDPOINT)
        headers: Dict[str, str] = {}
        headers.update(self._get_default_headers())
        with file_path.open("rb") as file:
            form_data = aiohttp.FormData()
            form_data.add_field("file", file, filename=file_path.name)
            form_data.add_field("purpose", file_purpose)
            form_data.add_field("meta", json.dumps(file_metadata))
            resp_bytes = await self._request(
                "POST",
                url,
                data=form_data,
                headers=headers,
                raise_for_status=True,
            )
        result = self._get_result_from_response_body(resp_bytes)
        return self._create_file_obj_from_dict(result)

    async def retrieve_file(self, file_id: str) -> RemoteFile:
        """Retrieve a file in AI Studio client by id."""
        self.ensure_not_closed()
        url = self._get_url(self._RETRIEVE_ENDPOINT).format(file_id=file_id)
        headers: Dict[str, str] = {}
        headers.update(self._get_default_headers())
        resp_bytes = await self._request(
            "GET",
            url,
            headers=headers,
            raise_for_status=True,
        )
        result = self._get_result_from_response_body(resp_bytes)
        return self._create_file_obj_from_dict(result)

    async def retrieve_file_contents(self, file_id: str) -> bytes:
        """Retrieve file content in AI Studio client by id."""
        self.ensure_not_closed()
        url = self._get_url(self._RETRIEVE_CONTENTS_ENDPOINT).format(file_id=file_id)
        headers: Dict[str, str] = {}
        headers.update(self._get_default_headers())
        resp_bytes = await self._request(
            "GET",
            url,
            headers=headers,
            raise_for_status=True,
        )
        return resp_bytes

    async def list_files(self) -> List[RemoteFile]:
        """List files in AI Studio client."""
        self.ensure_not_closed()
        url = self._get_url(self._LIST_ENDPOINT)
        headers: Dict[str, str] = {}
        headers.update(self._get_default_headers())
        resp_bytes = await self._request(
            "GET",
            url,
            headers=headers,
            raise_for_status=True,
        )
        result = self._get_result_from_response_body(resp_bytes)
        files: List[RemoteFile] = []
        for item in result:
            file = self._create_file_obj_from_dict(item)
            files.append(file)
        return files

    async def delete_file(self, file_id: str) -> None:
        raise TypeError(f"`{self.__class__.__name__}.{inspect.stack()[0][3]}` is not supported.")

    async def create_temporary_url(self, file_id: str, expire_after: float) -> str:
        url = self._get_url(self._RETRIEVE_ENDPOINT).format(file_id=file_id)
        headers: Dict[str, str] = {}
        headers.update(self._get_default_headers())
        resp_bytes = await self._request(
            "GET",
            url,
            params={"expirationInSeconds": expire_after},
            headers=headers,
            raise_for_status=True,
        )
        result = self._get_result_from_response_body(resp_bytes)
        return result["fileUrl"]

    async def close(self) -> None:
        if not self._closed:
            await self._session.close()

    def _create_aiohttp_session(self) -> aiohttp.ClientSession:
        return aiohttp.ClientSession(**self._get_session_config())

    def _get_session_config(self) -> Dict[str, Any]:
        return {}

    def _get_default_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"token {self._access_token}",
        }

    async def _request(self, *args: Any, **kwargs: Any) -> bytes:
        async with self._session.request(*args, **kwargs) as response:
            return await response.read()

    def _create_file_obj_from_dict(self, dict_: Dict[str, Any]) -> RemoteFile:
        metadata: Dict[str, Any]
        if dict_.get("meta"):
            metadata = json.loads(dict_["meta"])
            if not isinstance(metadata, dict):
                raise FileError(f"Invalid metadata: {dict_['meta']}")
        else:
            metadata = {}
        return RemoteFile(
            id=dict_["fileId"],
            filename=dict_["fileName"],
            byte_size=dict_["bytes"],
            created_at=dict_["createTime"],
            purpose=dict_["purpose"],
            metadata=metadata,
            client=self,
        )

    def _get_result_from_response_body(self, resp_body: bytes) -> Any:
        decoded_resp_body = resp_body.decode("utf-8")
        try:
            resp_dict = json.loads(decoded_resp_body)
        except json.JSONDecodeError:
            raise FileError(f"The response body is not valid JSON: {decoded_resp_body}")
        if not isinstance(resp_dict, dict):
            raise FileError(f"The response body can not be parsed as a dict: {decoded_resp_body}")
        if resp_dict.get("errorCode", -1) != 0:
            raise FileError(f"An error was encountered. Response body: {resp_dict}")
        if "result" not in resp_dict:
            raise FileError(f"The response body does not contain the 'result' key: {resp_dict}")
        return resp_dict["result"]

    @classmethod
    def _get_url(cls, path: str) -> str:
        base_url = os.getenv("AISTUDIO_BASE_URL", cls._BASE_URL)
        return f"{base_url}{path}"
