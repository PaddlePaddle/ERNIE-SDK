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

import unittest
from enum import Enum
from inspect import isclass
from typing import List, Optional

from erniebot_agent.tools.base import RemoteToolkit
from erniebot_agent.tools.schema import (
    ToolParameterView,
    get_typing_list_type,
    is_optional_type,
    json_type,
)
from erniebot_agent.utils.common import create_enum_class
from openapi_spec_validator.readers import read_from_filename
from pydantic import Field


class TestToolSchema(unittest.TestCase):
    openapi_file = "./tests/fixtures/openapi.yaml"

    def test_plugin_schema(self):
        schema = RemoteToolkit.from_openapi_file(self.openapi_file)

        self.assertEqual(schema.info.title, "单词本")
        self.assertEqual(schema.servers[0].url, "http://127.0.0.1:8081")

    def test_load_and_save(self):
        """function_call requires empty fields, eg: items: {}, but yaml file doesn't
        contain any empty field
        """
        spec_dict = read_from_filename(self.openapi_file)
        schema = RemoteToolkit.from_openapi_file(self.openapi_file)
        saved_spec_dict = schema.to_openapi_dict()
        self.assertEqual(spec_dict[0], saved_spec_dict)

    def test_function_call_schemas(self):
        toolkit = RemoteToolkit.from_openapi_file(self.openapi_file)
        function_call_schemas = [tool.function_call_schema() for tool in toolkit.get_tools()]
        self.assertEqual(len(function_call_schemas), 4)

        self.assertEqual(function_call_schemas[0]["name"], "getWordbook")
        self.assertEqual(function_call_schemas[0]["responses"]["required"], ["wordbook"])
        self.assertEqual(function_call_schemas[0]["responses"]["properties"]["wordbook"]["type"], "array")
        self.assertEqual(function_call_schemas[3]["name"], "deleteWord")

    def test_get_typing_list_type(self):
        result = get_typing_list_type(List[int])
        self.assertEqual(result, "integer")

        result = get_typing_list_type(List[str])
        self.assertEqual(result, "string")

        result = get_typing_list_type(int)
        self.assertEqual(result, None)

        result = get_typing_list_type(dict)
        self.assertEqual(result, None)

        result = get_typing_list_type(List[ToolParameterView])
        self.assertEqual(result, "object")

    def test_json_type(self):
        result = json_type(List[int])
        self.assertEqual(result, "array")

        result = json_type(int)
        self.assertEqual(result, "integer")

        result = json_type(float)
        self.assertEqual(result, "number")

        result = json_type(ToolParameterView)
        self.assertEqual(result, "object")

    def test_list_tool_parameter_view(self):
        class SearchResponseDocument(ToolParameterView):
            document: str = Field(description="和query相关的规章片段")
            filename: str = Field(description="规章名称")
            page_num: int = Field(description="规章页数")

        class SearchToolOutputView(ToolParameterView):
            documents: List[SearchResponseDocument] = Field(description="检索结果，内容为住房和城乡建设部规章中和query相关的规章片段")

        open_api_dict = SearchToolOutputView.to_openapi_dict()

        expected_schema = {
            "type": "object",
            "required": ["documents"],
            "properties": {
                "documents": {
                    "type": "array",
                    "description": "检索结果，内容为住房和城乡建设部规章中和query相关的规章片段",
                    "items": {
                        "type": "object",
                        "required": ["document", "filename", "page_num"],
                        "properties": {
                            "document": {"type": "string", "description": "和query相关的规章片段"},
                            "filename": {"type": "string", "description": "规章名称"},
                            "page_num": {"type": "integer", "description": "规章页数"},
                        },
                    },
                }
            },
        }
        self.assertDictEqual(open_api_dict, expected_schema)

    def test_optional_tool_parameter_view(self):
        class SearchResponseDocument(ToolParameterView):
            document: str = Field(description="和query相关的规章片段")
            filename: str = Field(description="规章名称")
            page_num: int = Field(description="规章页数")

        class SearchToolOutputView(ToolParameterView):
            name: str = Field(description="测试名称")
            documents: Optional[SearchResponseDocument] = Field(
                description="检索结果，内容为住房和城乡建设部规章中和query相关的规章片段", default_factory=list
            )

        openapi_dict = SearchToolOutputView.to_openapi_dict()
        expected_openapi_dict = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "测试名称",
                },
                "documents": {
                    "type": "object",
                    "description": "检索结果，内容为住房和城乡建设部规章中和query相关的规章片段",
                    "required": ["document", "filename", "page_num"],
                    "properties": {
                        "document": {
                            "type": "string",
                            "description": "和query相关的规章片段",
                        },
                        "filename": {
                            "type": "string",
                            "description": "规章名称",
                        },
                        "page_num": {
                            "type": "integer",
                            "description": "规章页数",
                        },
                    },
                },
            },
            "required": ["name"],
        }
        self.assertDictEqual(openapi_dict, expected_openapi_dict)

    def test_enum_value(self):
        # TODO(wj-Mcat): to support enum[int/str/float]
        pass

    def test_is_optional_type(self):
        self.assertFalse(is_optional_type(List[int]))
        self.assertFalse(is_optional_type(int))
        self.assertFalse(is_optional_type(str))

        self.assertTrue(is_optional_type(Optional[int]))
        self.assertTrue(is_optional_type(Optional[str]))
        self.assertTrue(is_optional_type(Optional[ToolParameterView]))

    def test_load_examples(self):
        toolkit = RemoteToolkit.from_openapi_file("./tests/fixtures/openapi.yaml")
        toolkit.examples = toolkit.load_examples_yaml("./tests/fixtures/examples.yaml")
        self.assertEqual(len(toolkit.examples), 10)

        # add_word examples
        examples = toolkit.get_tool("getWordbook").examples

        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0].content, "展示单词列表")
        self.assertEqual(examples[1].function_call["name"], "getWordbook")
        self.assertEqual(examples[1].function_call["thoughts"], "这是一个展示单词本的需求")

    def test_dynamic_enum_class(self):
        # 使用函数创建枚举类
        member_names = ["MEMBER1", "MEMBER2", "MEMBER3"]
        MyEnum = create_enum_class("MyEnum", member_names)

        self.assertTrue(issubclass(MyEnum, Enum))
        self.assertTrue(isclass(MyEnum))
        self.assertEqual(MyEnum.MEMBER1.value, "MEMBER1")
        self.assertListEqual(list(MyEnum.__members__.keys()), member_names)
