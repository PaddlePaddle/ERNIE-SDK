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

import os
import uuid
from typing import Dict, List, Optional, Type

from erniebot_agent.chat_models.erniebot import ERNIEBot
from erniebot_agent.messages import AIMessage, FunctionMessage, HumanMessage, Message
from erniebot_agent.prompt import PromptTemplate
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
                    "thoughts": "用户需要我生成一张小狗的图片，图像高度为512，宽度为512。我可以使用ImageGenerationTool工具来满足用户的需求。",
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
                    "并设置prompt为'生成两张天空的图片'，width和height可以默认为512，image_num默认为1。",
                    "arguments": '{"prompt":"生成两张天空的图片","width":512,"height":512,"image_num":1}',
                },
            ),
            HumanMessage("使用AI作图工具，生成1张小猫的图片，高度和高度是1024"),
            AIMessage(
                "",
                function_call={
                    "name": self.tool_name,
                    "thoughts": "用户需要生成一张小猫的图片，高度和宽度都是1024。我可以使用ImageGenerationTool工具来满足用户的需求。",
                    "arguments": '{"prompt":"生成一张小猫的照片。","width":1024,"height":1024,"image_num":1}',
                },
            ),
        ]


class PromptGeneratorInputView(ToolParameterView):
    input_prompt: str = Field(description="用户输入的prompt")


class PromptGeneratorOutputView(ToolParameterView):
    refined_prompt: float = Field(description="润色后的prompt")


class PromptGeneratorTool(Tool):
    description: str = "提示词生成器根据你提供的主题或简单Prompt，快速丰富、扩展、优化为高质量的Prompt，帮你写出最“专业”的Prompt。"
    input_type: Type[ToolParameterView] = PromptGeneratorInputView
    ouptut_type: Type[ToolParameterView] = PromptGeneratorOutputView

    def __init__(self, model: ERNIEBot):
        template = """根据输入的内容重新生成一组简洁明了的可以被用于生成图片的描述词。包含具体的主体画面信息，主体画面信息要求简单，不能复杂。生成的内容满足以下要求：
        1. 生成的内容中不能出现英文，全文不允许使用任何形式的序号和书面化符号，如括号和冒号，不要出现和输入内容完全一致的信息;
        2. 输出内容的长度保持在50个字以内，输出内容需要通过“在什么场景下，什么人在什么时间点干什么事情”，对于中国传统故事，需要简短概括表达，最好不要用原始的输入表达。
        样例：
        例子1：输入内容：车水马龙。输出内容：交通拥挤的古风街道
        例子2：输入内容：鹅毛大雪。输出内容：冬天，雪花，森林
        例子3：输入内容：应有尽有。输出内容：丰富的商品，丰富的物品
        例子4：输入内容：百年好合。输出内容：一对夫妻，坐在一起，眺望远方
        例子5：输入内容：刻舟求剑。输出内容：一只木船在河流上，一个小孩和老人站在船上
        例子7：输入内容：两小无猜。输出内容：男孩和女孩在玩耍
        例子8：输入内容：熊熊大火。输出内容：山林大火
        例子9：输入内容：人山人海。输出内容：很多人，人流拥挤
        例子10：输入内容：牛逼哄哄。输出内容：戴墨镜金链子的大哥，霸气
        例子11：输入内容：桃李满天下。输出内容：教师在给学生上课
        例子12：输入内容：固若金汤。输出内容：华夏古代城墙
        输入内容：{{input_prompt}}
        输出内容："""
        self.prompt_template = PromptTemplate(template=template, input_variables=["input_prompt"])
        self.model = model

    async def __call__(self, input_prompt: str) -> Dict[str, str]:
        prompt = self.prompt_template.format(input_prompt=input_prompt)
        response = await self.model.async_chat([HumanMessage(content=prompt)], stream=False)
        return {"refined_prompt": response.content}

    @property
    def examples(self) -> List[Message]:
        return [
            HumanMessage("画一张小狗飞奔的画"),
            AIMessage(
                "",
                function_call={
                    "name": self.tool_name,
                    "thoughts": (
                        f"用户想画一张小狗飞奔的画，我先使用{self.tool_name}工具进行prompt优化，其中`input_prompt`字段设置为：'小狗飞奔',"
                        "接着再用ImageGenerationTool生成图片。"
                    ),
                    "arguments": '{"input_prompt": "小狗飞奔"}',
                },
            ),
            FunctionMessage(
                name=self.tool_name, content='{"refined_prompt":"在乡村小路上，一只小狗在奔跑，周围是一片翠绿的田野和稀疏的房屋"'
            ),
            AIMessage(
                "",
                function_call={
                    "name": self.tool_name,
                    "thoughts": (
                        "用户想画一张小狗飞奔的画，我已完成prompt优化，接下来使用ImageGenerationTool生成图片，"
                        "将prompt设置为'在乡村小路上，一只小狗在奔跑，周围是一片翠绿的田野和稀疏的房屋'，"
                        "width和height可以默认为512，image_num默认为1。"
                    ),
                    "arguments": (
                        '{"prompt": "在乡村小路上，一只小狗在奔跑，周围是一片翠绿的田野和稀疏的房屋", '
                        '"width":512,"height":512,"image_num":1}'
                    ),
                },
            ),
        ]
