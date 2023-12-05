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
from __future__ import annotations

import asyncio
import os
import socket
import tempfile
import time
import unittest
import uuid

import uvicorn
from erniebot_agent.file_io.file_manager import FileManager
from erniebot_agent.tools.base import RemoteToolkit
from fastapi import FastAPI
from fastapi.responses import FileResponse

PYYAML_CONTENT = """openapi: 3.0.1
info:
    title: TestRemoteTool
    description: 个性化的英文单词本，可以增加、删除和浏览单词本中的单词，背单词时从已有单词本中随机抽取单词生成句子或者段落。
    version: "v1"
servers:
    - url: http://0.0.0.0:<<port>>
paths:
    /get_file:
        get:
            operationId: getFile
            description: 展示单词列表
            requestBody:
                content:
                    application/json:
                        schema:
                            $ref: "#/components/schemas/getFileRequest"
            responses:
                "200":
                    description: 列表展示完成
                    content:
                        application/json:
                            schema:
                                $ref: "#/components/schemas/getFileResponse"

components:
    schemas:
        getFileRequest:
            type: object
            required: [file]
            properties:
                file:
                    type: string
                    format: byte
                    description: 文件的 ID
            x-erniebot-agent-file:
                - file

        getFileResponse:
            type: object
            required: [response_file]
            properties:
                response_file:
                    type: string
                    format: byte
                    description: 单词本单词列表
            x-erniebot-agent-file:
                - response_file
"""


def start_local_file_server(port, file_path, file_name):
    app = FastAPI()

    @app.get(
        "/get_file",
        response_class=FileResponse,
    )
    def get_file():
        return FileResponse(file_path, filename=file_name)

    uvicorn.run(app, host="0.0.0.0", port=port)


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.01)
        try:
            s.bind(("localhost", port))
            return False
        except socket.error:
            return True


class TestToolWithFile(unittest.TestCase):
    def avaliable_free_port(self, exclude=None):
        exclude = exclude or []
        for port in range(8000, 9000):
            if port in exclude:
                continue
            if is_port_in_use(port):
                continue
            return port

        raise ValueError("can not get valiable port in [8000, 8200]")

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.file_name = "file.npy"
        self.file_path = os.path.join(self.tempdir.name, self.file_name)
        self.content = str(uuid.uuid4())
        with open(self.file_path, "w", encoding="utf-8") as f:
            f.write(self.content)

        from multiprocessing import Process

        self.port = self.avaliable_free_port()
        p = Process(target=start_local_file_server, args=(self.port, self.file_path, self.file_name))
        p.daemon = True
        p.start()

    def wait_until_server_is_ready(self):
        while True:
            if is_port_in_use(self.port):
                break

            print("waiting for server ...")
            time.sleep(1)

    def test_plugin_schema(self):
        self.wait_until_server_is_ready()
        with tempfile.TemporaryDirectory() as tempdir:
            openapi_file = os.path.join(tempdir, "openapi.yaml")
            content = PYYAML_CONTENT.replace("<<port>>", str(self.port))
            with open(openapi_file, "w", encoding="utf-8") as f:
                f.write(content)

            toolkit = RemoteToolkit.from_openapi_file(openapi_file)
            tool = toolkit.get_tool("getFile")
            # Check if tool_name_prefix in RemoteToolkit is properly prepended
            self.assertEqual(tool.tool_name, "TestRemoteTool/v1/getFile")
            file_manager = FileManager()
            input_file = asyncio.run(file_manager.create_file_from_path(self.file_path))
            result = asyncio.run(tool(file=input_file.id))
            self.assertIn("response_file", result)
            file_id = result["response_file"]

            file = file_manager.look_up_file_by_id(file_id=file_id)
            content = asyncio.run(file.read_contents())
            self.assertEqual(content.decode("utf-8"), self.content)
