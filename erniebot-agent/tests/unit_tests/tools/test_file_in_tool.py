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

import base64
import json
import os
import socket
import tempfile
import time
import unittest
import uuid

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from requests.models import Response

from erniebot_agent.file import FileManager
from erniebot_agent.tools import RemoteToolkit
from erniebot_agent.tools.utils import (
    get_file_info_from_param_view,
    parse_file_from_json_response,
)

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
            required: [file]
            properties:
                file:
                    type: string
                    format: byte
                    description: 单词本单词列表
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


class TestToolWithFile(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.file_manager = FileManager()

    async def asyncTearDown(self):
        await self.file_manager.close()

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

    async def test_plugin_schema(self):
        self.wait_until_server_is_ready()
        with tempfile.TemporaryDirectory() as tempdir:
            openapi_file = os.path.join(tempdir, "openapi.yaml")
            content = PYYAML_CONTENT.replace("<<port>>", str(self.port))
            with open(openapi_file, "w", encoding="utf-8") as f:
                f.write(content)

            toolkit = RemoteToolkit.from_openapi_file(openapi_file, file_manager=self.file_manager)
            tool = toolkit.get_tool("getFile")
            # tool.tool_name should have `tool_name_prefix`` prepended
            self.assertEqual(tool.tool_name, "TestRemoteTool/v1/getFile")
            input_file = await self.file_manager.create_file_from_path(self.file_path)
            result = await tool(file=input_file.id)
            self.assertIn("file", result)
            file_id = result["file"]

            file = self.file_manager.look_up_file_by_id(file_id=file_id)
            content = await file.read_contents()
            self.assertEqual(content.decode("utf-8"), self.content)
            self.assertIn("prompt", result)


class TestPlainJsonFileParser(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.file_manager = FileManager()

    async def asyncTearDown(self):
        await self.file_manager.close()

    def create_fake_response(self, body: dict):
        the_response = Response()
        the_response.code = "expired"
        the_response.error_type = "expired"
        the_response.status_code = 200
        the_response._content = json.dumps(body).encode("utf-8")
        the_response.headers["Content-Type"] = "application/json"
        return the_response

    async def test_plain_file(self):
        body = {"error_code": 0, "b64_img": str(base64.b64encode(b"sodisodjsodjsdoijisodfj"))}
        yaml_content = """openapi: 3.0.1
info:
  title: 文生图
  description: 文生图 v2.0
  version: v2.0
servers:
- url: http://tool-text-to-image.sandbox-aistudio-hub.baidu.com
  description: 文生图
paths:
  /text2image:
    post:
      summary: 文生图
      description: 文生图
      operationId: text2image
      parameters: []
      responses:
        "200":
          description: response success
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/text2imageResponse'
components:
  schemas:
    text2imageResponse:
      type: object
      description: "文生图的返回内容"
      properties:
        error_code:
            type: integer
            description: 请求返回的错误吗
        b64_img:
            type: string
            format: byte
            description: "图片的base64编码数据"
            x-ebagent-file-mime-type: image/png
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "openapi.yaml")
            with open(file_path, "w+", encoding="utf-8") as f:
                f.write(yaml_content)
            toolkit = RemoteToolkit.from_openapi_file(file_path, file_manager=self.file_manager)
            response = self.create_fake_response(body)
            tool = toolkit.get_tools()[-1]

            file_infos = get_file_info_from_param_view(tool.tool_view.returns)
            assert len(file_infos) == 1

            json_response = await parse_file_from_json_response(
                response.json(),
                file_manager=self.file_manager,
                param_view=tool.tool_view.returns,
                tool_name=tool.tool_name,
            )
            assert json_response["b64_img"].startswith("file-local-")


class TestJsonNestFileParser(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.file_manager = FileManager()

    async def asyncTearDown(self):
        await self.file_manager.close()

    def create_fake_response(self, body: dict):
        the_response = Response()
        the_response.code = "expired"
        the_response.error_type = "expired"
        the_response.status_code = 200
        the_response._content = json.dumps(body).encode("utf-8")
        the_response.headers["Content-Type"] = "application/json"
        return the_response

    async def test_plain_file(self):
        body = {"error_code": 0, "image": {"b64_img": str(base64.b64encode(b"sodisodjsodjsdoijisodfj"))}}
        yaml_content = """openapi: 3.0.1
info:
  title: 文生图
  description: 文生图 v2.0
  version: v2.0
servers:
- url: http://tool-text-to-image.sandbox-aistudio-hub.baidu.com
  description: 文生图
paths:
  /text2image:
    post:
      summary: 文生图
      description: 文生图
      operationId: text2image
      parameters: []
      responses:
        "200":
          description: response success
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/text2imageResponse'
components:
  schemas:
    text2imageResponse:
      type: object
      description: "文生图的返回内容"
      properties:
        error_code:
          type: integer
          description: 请求返回的错误吗
        image:
          type: object
          description: 图片内容信息
          properties:
            b64_img:
              type: string
              description: 图片的base64编码数据
              format: byte
              x-ebagent-file-mime-type: image/png
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "openapi.yaml")
            with open(file_path, "w+", encoding="utf-8") as f:
                f.write(yaml_content)
            toolkit = RemoteToolkit.from_openapi_file(file_path, file_manager=self.file_manager)
            response = self.create_fake_response(body)
            tool = toolkit.get_tools()[-1]

            file_infos = get_file_info_from_param_view(tool.tool_view.returns)
            assert len(file_infos) == 1

            assert file_infos["image"]["b64_img"]["format"] == "byte"

            json_response = await parse_file_from_json_response(
                response.json(),
                file_manager=self.file_manager,
                param_view=tool.tool_view.returns,
                tool_name=tool.tool_name,
            )
            assert json_response["image"]["b64_img"].startswith("file-local-")


class TestJsonNestListFileParser(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.file_manager = FileManager()

    async def asyncTearDown(self):
        await self.file_manager.close()

    def create_fake_response(self, body: dict):
        the_response = Response()
        the_response.code = "expired"
        the_response.error_type = "expired"
        the_response.status_code = 200
        the_response._content = json.dumps(body).encode("utf-8")
        the_response.headers["Content-Type"] = "application/json"
        return the_response

    async def test_plain_file(self):
        body = {"error_code": 0, "data": [{"b64_img": str(base64.b64encode(b"sodisodjsodjsdoijisodfj"))}]}
        yaml_content = """openapi: 3.0.1
info:
  title: 文生图
  description: 文生图 v2.0
  version: v2.0
servers:
- url: http://tool-text-to-image.sandbox-aistudio-hub.baidu.com
  description: 文生图
paths:
  /text2image:
    post:
      summary: 文生图
      description: 文生图
      operationId: text2image
      parameters: []
      responses:
        "200":
          description: response success
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/text2imageResponse'
components:
  schemas:
    text2imageResponse:
      type: object
      description: "文生图的返回内容"
      properties:
        error_code:
          type: integer
          description: 请求返回的错误吗
        data:
          type: array
          description: 图片内容列表信息
          items:
            type: object
            description: 图片的列表内容实体
            properties:
                b64_img:
                    type: string
                    description: 图片的base64编码数据
                    format: byte
                    x-ebagent-file-mime-type: image/png
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "openapi.yaml")
            with open(file_path, "w+", encoding="utf-8") as f:
                f.write(yaml_content)
            toolkit = RemoteToolkit.from_openapi_file(file_path, file_manager=self.file_manager)
            response = self.create_fake_response(body)
            tool = toolkit.get_tools()[-1]

            file_infos = get_file_info_from_param_view(tool.tool_view.returns)
            assert len(file_infos) == 1
            assert file_infos["data"]["b64_img"]["format"] == "byte"

            json_response = await parse_file_from_json_response(
                response.json(),
                file_manager=self.file_manager,
                param_view=tool.tool_view.returns,
                tool_name=tool.tool_name,
            )
            assert json_response["data"][0]["b64_img"].startswith("file-local-")
