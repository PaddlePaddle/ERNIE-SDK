from __future__ import annotations

import os
import uuid
from typing import Dict, List, Optional, Type

from erniebot_agent.messages import AIMessage, HumanMessage, Message
from erniebot_agent.tools.base import Tool
from erniebot_agent.tools.schema import ToolParameterView
from erniebot_agent.utils.common import download_file, get_cache_dir
from pydantic import Field

import erniebot


class ImageGenerationInputView(ToolParameterView):
    prompt: str = Field(description="描述图像内容、风格的文本。例如：生成一张月亮的照片，月亮很圆。")
    width: Optional[int] = Field(description="生成图片的宽度")
    height: Optional[int] = Field(description="生成图片的高度")
    image_num: Optional[int] = Field(description="生成图片的数量")


class ImageGenerationOutputView(ToolParameterView):
    image_path: str = Field(description="图片在本地机器上的保存路径")


class ImageGenerationTool(Tool):
    description: str = "AI作图、生成图片、画图的工具"
    input_type: Type[ToolParameterView] = ImageGenerationInputView
    ouptut_type: Type[ToolParameterView] = ImageGenerationOutputView

    def __init__(
        self,
        yinian_access_token: Optional[str] = None,
        yinian_ak: Optional[str] = None,
        yinian_sk: Optional[str] = None,
    ) -> None:
        if yinian_access_token is not None:
            self.config = {"api_type": "yinian", "access_token": yinian_access_token}
        elif yinian_ak is not None and yinian_sk is not None:
            self.config = {"api_type": "yinian", "ak": yinian_ak, "sk": yinian_sk}
        else:
            raise ValueError("Please set the yinian_access_token, or set yinian_ak and yinian_sk")

    async def __call__(
        self,
        prompt: str,
        width: Optional[int] = 512,
        height: Optional[int] = 512,
        image_num: Optional[int] = 1,
    ) -> Dict[str, List[str]]:
        response = erniebot.Image.create(
            model="ernie-vilg-v2",
            prompt=prompt,
            width=width,
            height=height,
            image_num=image_num,
            _config_=self.config,
        )

        image_path = []
        cache_dir = get_cache_dir()
        for item in response["data"]["sub_task_result_list"]:
            image_url = item["final_image_list"][0]["img_url"]
            save_path = os.path.join(cache_dir, f"img_{uuid.uuid1()}.png")
            download_file(image_url, save_path)
            image_path.append(save_path)
        return {"image_path": image_path}

    @property
    def examples(self) -> List[Message]:
        return [
            HumanMessage("画一张小狗的图片，图像高度512，图像宽度512"),
            AIMessage(
                "",
                function_call={
                    "name": "ImageGenerationTool",
                    "thoughts": "用户需要我生成一张小狗的图片，图像高度为512，宽度为512。" "我可以使用ImageGenerationTool工具来满足用户的需求。",
                    "arguments": '{"prompt":"画一张小狗的图片，图像高度512，图像宽度512",'
                    '"width":512,"height":512,"image_num":1}',
                },
            ),
            HumanMessage("生成两张天空的图片"),
            AIMessage(
                "",
                function_call={
                    "name": self.tool_name,
                    "thoughts": "用户想要生成两张天空的图片，我需要调用ImageGenerationTool工具的call接口，"
                    "并设置prompt为'生成两张天空的图片'，width和height可以默认为512，image_num默认为2。",
                    "arguments": '{"prompt":"生成两张天空的图片","width":512,"height":512,"image_num":2}',
                },
            ),
            HumanMessage("使用AI作图工具，生成1张小猫的图片，高度和高度是1024"),
            AIMessage(
                "",
                function_call={
                    "name": self.tool_name,
                    "thoughts": "用户需要生成一张小猫的图片，高度和宽度都是1024。" "我可以使用ImageGenerationTool工具来满足用户的需求。",
                    "arguments": '{"prompt":"生成一张小猫的照片。","width":1024,"height":1024,"image_num":1}',
                },
            ),
        ]
