from __future__ import annotations

import json

import pytest

from erniebot_agent.tools.remote_toolkit import RemoteToolkit

from .base import RemoteToolTesting


class TestPPRemoteTool(RemoteToolTesting):
    @pytest.mark.asyncio
    async def test_pp_matting(self):
        toolkit = RemoteToolkit.from_aistudio("pp-matting", file_manager=self.file_manager)

        file = await self.file_manager.create_file_from_path(self.download_fixture_file("trans.png"))
        agent = self.get_agent(toolkit)

        result = await agent.run("请帮我对图片中的人像抠出来", files=[file])

        self.assertEqual(len(result.files), 2)

    @pytest.mark.asyncio
    async def test_pp_human_v2(self):
        toolkit = RemoteToolkit.from_aistudio("pp-human-v2", file_manager=self.file_manager)

        file = await self.file_manager.create_file_from_path(self.download_fixture_file("human_attr.jpg"))
        agent = self.get_agent(toolkit)

        result = await agent.run("请帮我对图中的行人进行分析", files=[file])
        self.assertEqual(result.files[-1].type, "output")

    @pytest.mark.asyncio
    async def test_pp_humansegv2(self):
        toolkit = RemoteToolkit.from_aistudio("humanseg", file_manager=self.file_manager)

        agent = self.get_agent(toolkit)

        file_path = self.download_fixture_file("humanseg_input_img.jpg")
        file = await self.file_manager.create_file_from_path(file_path)

        result = await agent.run("对这张图片进行人像分割，包含的文件为：", files=[file])
        self.assertEqual(len(result.files), 2)
        self.assertEqual(len(result.steps), 1)

    @pytest.mark.asyncio
    async def test_pp_tinypose(self):
        toolkit = RemoteToolkit.from_aistudio("pp-tinypose", file_manager=self.file_manager)
        agent = self.get_agent(toolkit)

        file_path = self.download_fixture_file("pp_tinypose_input_img.jpg")
        file = await self.file_manager.create_file_from_path(file_path)

        result = await agent.run("检测这张图片中的人体关键点，包含的文件为：", files=[file])
        self.assertEqual(len(result.files), 2)
        self.assertEqual(len(result.steps), 1)

    @pytest.mark.asyncio
    async def test_pp_vehicle(self):
        toolkit = RemoteToolkit.from_aistudio("pp-vehicle", file_manager=self.file_manager)

        file = await self.file_manager.create_file_from_path(self.download_fixture_file("vehicle.jpg"))
        agent = self.get_agent(toolkit)

        response = await agent.run("请帮我对这幅图进行交通场景分析", files=[file])

        expected_tool_name = toolkit.get_tool("analyzeVehicles").tool_name
        self.assertEqual(response.chat_history[2].name, expected_tool_name)
        decoded_tool_ret = json.loads(response.chat_history[2].content)
        self.assertIn("vehicle_plates", decoded_tool_ret)
        self.assertEqual(decoded_tool_ret["vehicle_plates"], ["CCL9542"])
        self.assertIn("vehicle_attrs", decoded_tool_ret)
        self.assertEqual(decoded_tool_ret["vehicle_attrs"], [{"color": "blue", "kind": "Unknown"}])
        self.assertEqual(len(response.files), 2)

    @pytest.mark.asyncio
    async def test_pp_structure(self):
        toolkit = RemoteToolkit.from_aistudio("pp-structure-v2", file_manager=self.file_manager)

        agent = self.get_agent(toolkit)
        file = await self.file_manager.create_file_from_path(self.download_fixture_file("ocr_table.png"))
        result = await agent.run("请帮我提取一下这个表格的内容", files=[file])
        self.assertEqual(len(result.files), 1)
        self.assertIn("设备", result.text)

    @pytest.mark.asyncio
    async def test_pp_ocr_v4(self):
        toolkit = RemoteToolkit.from_aistudio("pp-ocrv4", file_manager=self.file_manager)

        file = await self.file_manager.create_file_from_path(
            self.download_fixture_file("ocr_example_input.png")
        )
        agent = self.get_agent(toolkit)

        response = await agent.run("请帮我识别出这幅图片中的文字", files=[file])

        self.assertEqual(len(response.steps), 1)
        decoded_tool_ret = json.loads(response.chat_history[2].content)
        self.assertEqual(decoded_tool_ret, {"result": "中国\n汉字"})
