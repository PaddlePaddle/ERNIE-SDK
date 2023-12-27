from typing import Type

from pydantic import Field

from erniebot_agent.file import GlobalFileManagerHandler
from erniebot_agent.tools.base import Tool, ToolParameterView


class ImageGenerateToolInputView(ToolParameterView):
    query: str = Field(description="场景图片的描述")


class ImageGenerateToolOutputView(ToolParameterView):
    output_image: str = Field(description="返回的图片文件，格式为file-xxxx, 不包括<file></file>")


class ImageGenerateTool(Tool):
    description: str = "根据当前的互动内容，按照场景图片部分的内容生成图片。"
    input_type: Type[ToolParameterView] = ImageGenerateToolInputView
    otuput_story: Type[ToolParameterView] = ImageGenerateToolOutputView

    async def __call__(self, query: str) -> str:
        # output_dir = query
        file_manager = await GlobalFileManagerHandler().get()
        # self.file_manager.create_file_from_bytes()
        file = await file_manager.create_file_from_path("/Users/tanzhehao/Desktop/git.png")  # for mimic
        return {"output_image": file.id}
