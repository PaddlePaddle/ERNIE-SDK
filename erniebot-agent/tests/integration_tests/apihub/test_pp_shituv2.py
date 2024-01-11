# from __future__ import annotations

# import pytest

# from erniebot_agent.file import FileManager
# from erniebot_agent.tools import RemoteToolkit

# from .base import RemoteToolTesting


# class TestRemoteTool(RemoteToolTesting):
#     @pytest.mark.asyncio
#     async def test_pp_shituv2(self):
#         toolkit = RemoteToolkit.from_aistudio("pp-shituv2", file_manager=self.file_manager)
#         tools = toolkit.get_tools()
#         print(tools[0].function_call_schema())

#         agent = self.get_agent(toolkit)

#         async with FileManager() as file_manager:
#             file_path = self.download_file(
#                 "https://paddlenlp.bj.bcebos.com/ebagent/ci/fixtures/remote-tools/pp_shituv2_input_img.png"
#             )
#             file = await file_manager.create_file_from_path(file_path)

#             result = await agent.run("对这张图片进行通用识别，包含的文件为：", files=[file])
#             files = self.get_files(result)
#             action_steps = self.get_action_steps(result)
#             self.assertEqual(len(files), 2)
#             self.assertEqual(len(action_steps), 1)
#             self.assertIn("file-", result.text)
