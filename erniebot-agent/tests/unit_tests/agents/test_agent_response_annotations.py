import asyncio
import unittest
from typing import List, Literal

from erniebot_agent.agents.schema import AgentFile, AgentResponse
from erniebot_agent.file_io import get_file_manager


class TestAgentResponseAnnotations(unittest.TestCase):
    def setUp(self):
        self.test = ""
        self.file_manager = get_file_manager()
        self.file1 = asyncio.run(self.file_manager.create_file_from_bytes(b"test1", "test1.txt"))
        self.file2 = asyncio.run(self.file_manager.create_file_from_bytes(b"test2", "test2.txt"))

    def test_agent_response_onefile_oneagentfile(self):
        agent_file = AgentFile(file=self.file1, type="input", used_by="")

        self.files: List["AgentFile"] = [agent_file]

        text = f"根据您的要求，我已经将图像中的汽车分割出来，生成的图像文件为{self.file1.id}，您可以保存该文件进行查看。如果您还有其他问题或需要进一步的操作，请随时告诉我。"

        agent_response = AgentResponse(
            files=self.files, text=text, chat_history=[], actions=[], status=Literal["FINISHED"]
        )
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
        agent_file1 = AgentFile(file=self.file1, type="input", used_by="")
        agent_file2 = AgentFile(file=self.file2, type="input", used_by="")

        self.files: List["AgentFile"] = [agent_file1, agent_file2]

        text = f"根据您的要求，我已经将图像中的汽车分割出来，生成的图像文件为{self.file1.id}，您可以保存该文件进行查看。如果您还有其他问题或需要进一步的操作，请随时告诉我。"

        agent_response = AgentResponse(
            files=self.files, text=text, chat_history=[], actions=[], status=Literal["FINISHED"]
        )
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
        agent_file = AgentFile(file=self.file1, type="input", used_by="")

        self.files: List["AgentFile"] = [agent_file]

        text = "根据您的要求，我已经将图像中的汽车分割出来，您可以保存该文件进行查看。如果您还有其他问题或需要进一步的操作，请随时告诉我。"

        agent_response = AgentResponse(
            files=self.files, text=text, chat_history=[], actions=[], status=Literal["FINISHED"]
        )
        self.assertEqual(
            agent_response.annotations,
            {"content_parts": [{"text": "根据您的要求，我已经将图像中的汽车分割出来，您可以保存该文件进行查看。如果您还有其他问题或需要进一步的操作，请随时告诉我。"}]},
        )


if __name__ == "__main__":
    unittest.main()
