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
import pathlib
from typing import Any, ClassVar, Dict, List, Optional

import aiohttp
from erniebot_agent.file_io.base import File
from erniebot_agent.file_io.protocol import FilePurpose, is_remote_file_id


class RemoteFile(File):
    def __init__(
        self,
        *,
        id: str,
        filename: str,
        byte_size: int,
        created_at: int,
        purpose: FilePurpose,
        metadata: Dict[str, Any],
        client: "RemoteFileClient",
    ) -> None:
        if not is_remote_file_id(id):
            raise ValueError(f"Invalid file ID: {id}")
        super().__init__(
            id=id,
            filename=filename,
            byte_size=byte_size,
            created_at=created_at,
            purpose=purpose,
            metadata=metadata,
        )
        self._client = client

    async def read_contents(self) -> bytes:
        file_contents = await self._client.retrieve_file_contents(self.id)
        return file_contents

    async def delete(self) -> None:
        await self._client.delete_file(self.id)

    def file_repr_with_URL(self) -> str:
        self.URL = self._get_url()
        return f"<file>{self.id}</file><url>{self.URL}</url>"

    def _get_url(self) -> str:
        """Get URL from AiStudio."""
        # TODO(shiyutang): Get URL from AiStudio.
        return """https://qianfan-doc.bj.bcebos.com/chatfile/\
%E6%B5%85%E8%B0%88%E7%89%9B%E5%A5%B6%E7%9A%84%\
E8%90%A5%E5%85%BB%E4%B8%8E%E6%B6%88%E8%B4%B9%E8%B6%8B%E5%8A%BF.docx"""


class RemoteFileClient(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    async def upload_file(
        self, file_path: pathlib.Path, file_purpose: FilePurpose, file_metadata: Dict[str, Any]
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


class AIStudioFileClient(RemoteFileClient):
    _BASE_URL: ClassVar[str] = "https://sandbox-aistudio.baidu.com"
    _UPLOAD_ENDPOINT: ClassVar[str] = "/llm/lmapp/files"
    _RETRIEVE_ENDPOINT: ClassVar[str] = "/llm/lmapp/files/{file_id}"
    _RETRIEVE_CONTENTS_ENDPOINT: ClassVar[str] = "/llm/lmapp/files/{file_id}/content"
    _LIST_ENDPOINT: ClassVar[str] = "/llm/lmapp/files"

    def __init__(
        self, access_token: str, *, aiohttp_session: Optional[aiohttp.ClientSession] = None
    ) -> None:
        super().__init__()
        self._access_token = access_token
        self._session = aiohttp_session

    async def upload_file(
        self, file_path: pathlib.Path, file_purpose: FilePurpose, file_metadata: Dict[str, Any]
    ) -> RemoteFile:
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
        return self._build_file_obj_from_dict(result)

    async def retrieve_file(self, file_id: str) -> RemoteFile:
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
        return self._build_file_obj_from_dict(result)

    async def retrieve_file_contents(self, file_id: str) -> bytes:
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
            file = self._build_file_obj_from_dict(item)
            files.append(file)
        return files

    async def delete_file(self, file_id: str) -> None:
        raise RuntimeError(f"`{self.__class__.__name__}.{inspect.stack()[0][3]}` is not supported.")

    async def _request(self, *args: Any, **kwargs: Any) -> bytes:
        if self._session is not None:
            async with self._session.request(*args, **kwargs) as response:
                return await response.read()
        else:
            async with aiohttp.ClientSession(**self._get_session_config()) as session:
                async with session.request(*args, **kwargs) as response:
                    return await response.read()

    def _get_session_config(self) -> Dict[str, Any]:
        return {}

    def _get_default_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"token {self._access_token}",
        }

    def _build_file_obj_from_dict(self, dict_: Dict[str, Any]) -> RemoteFile:
        metadata: Dict[str, Any]
        if "meta" in dict_:
            metadata = json.loads(dict_["meta"])
            if not isinstance(metadata, dict):
                raise ValueError(f"Invalid metadata: {dict_['meta']}")
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
            raise RuntimeError(f"The response body is not valid JSON: {decoded_resp_body}")
        if not isinstance(resp_dict, dict):
            raise RuntimeError(f"The response body can not be parsed as a dict: {decoded_resp_body}")
        if resp_dict.get("errorCode", -1) != 0:
            raise RuntimeError(f"An error was encountered. Response body: {resp_dict}")
        if "result" not in resp_dict:
            raise RuntimeError(f"The response body does not contain the 'result' key: {resp_dict}")
        return resp_dict["result"]

    @classmethod
    def _get_url(cls, path: str) -> str:
        return f"{cls._BASE_URL}{path}"
