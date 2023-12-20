import asyncio
import unittest
from typing import List, Literal

from erniebot_agent.agents.schema import AgentResponse, ToolInfo, ToolStep
from erniebot_agent.file_io import File, get_file_manager


class TestAgentResponseAnnotations(unittest.TestCase):
    def setUp(self):
        self.test = ""
        self.file_manager = get_file_manager()
        self.file1 = asyncio.run(self.file_manager.create_file_from_bytes(b"test1", "test1.txt"))
        self.file2 = asyncio.run(self.file_manager.create_file_from_bytes(b"test2", "test2.txt"))

    def test_agent_response_onefile_oneagentfile(self):
        self.files: List[File] = [self.file1]

        text = f"根据您的要求，我已经将图像中的汽车分割出来，生成的图像文件为{self.file1.id}，您可以保存该文件进行查看。如果您还有其他问题或需要进一步的操作，请随时告诉我。"
        step = (
            ToolStep(
                info=ToolInfo(tool_name="", tool_args=""),
                result="",
                input_files=[],
                output_files=self.files,
            ),
        )
        agent_response = AgentResponse(text=text, chat_history=[], steps=step, status=Literal["FINISHED"])
        self.assertEqual(
            agent_response.annotations,
            {
                "content_parts": [
                    {"text": "根据您的要求，我已经将图像中的汽车分割出来，生成的图像文件为"},
                    {
                        "id": self.file1.id,
                        "filename": self.file1.filename,
                        "byte_size": self.file1.byte_size,
                        "created_at": self.file1.created_at,
                        "purpose": self.file1.purpose,
                        "metadata": self.file1.metadata,
                    },
                    {"text": "，您可以保存该文件进行查看。如果您还有其他问题或需要进一步的操作，请随时告诉我。"},
                ]
            },
        )

    def test_agent_response_onefile_twoagentfile(self):
        text = f"根据您的要求，我已经将图像中的汽车分割出来，生成的图像文件为{self.file1.id}，您可以保存该文件进行查看。如果您还有其他问题或需要进一步的操作，请随时告诉我。"

        step = (
            ToolStep(
                info=ToolInfo(tool_name="", tool_args=""),
                result="",
                input_files=[self.file1],
                output_files=[self.file2],
            ),
        )
        agent_response = AgentResponse(text=text, chat_history=[], steps=step, status=Literal["FINISHED"])
        self.assertEqual(
            agent_response.annotations,
            {
                "content_parts": [
                    {"text": "根据您的要求，我已经将图像中的汽车分割出来，生成的图像文件为"},
                    {
                        "id": self.file1.id,
                        "filename": self.file1.filename,
                        "byte_size": self.file1.byte_size,
                        "created_at": self.file1.created_at,
                        "purpose": self.file1.purpose,
                        "metadata": self.file1.metadata,
                    },
                    {"text": "，您可以保存该文件进行查看。如果您还有其他问题或需要进一步的操作，请随时告诉我。"},
                ]
            },
        )

    def test_agent_response_nofile_twoagentfile(self):
        text = "根据您的要求，我已经将图像中的汽车分割出来，您可以保存该文件进行查看。如果您还有其他问题或需要进一步的操作，请随时告诉我。"
        step = (
            ToolStep(
                info=ToolInfo(tool_name="", tool_args=""),
                result="",
                input_files=[self.file1],
                output_files=[self.file2],
            ),
        )
        agent_response = AgentResponse(text=text, chat_history=[], steps=step, status=Literal["FINISHED"])

        self.assertEqual(
            agent_response.annotations,
            {"content_parts": [{"text": "根据您的要求，我已经将图像中的汽车分割出来，您可以保存该文件进行查看。如果您还有其他问题或需要进一步的操作，请随时告诉我。"}]},
        )


if __name__ == "__main__":
    unittest.main()
