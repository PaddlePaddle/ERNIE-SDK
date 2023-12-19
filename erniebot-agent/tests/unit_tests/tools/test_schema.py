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

from openapi_spec_validator.readers import read_from_filename
from pydantic import Field

from erniebot_agent.tools import RemoteToolkit
from erniebot_agent.tools.base import tool_response_contains_file
from erniebot_agent.tools.schema import (
    ToolParameterView,
    get_typing_list_type,
    is_optional_type,
    json_type,
)
from erniebot_agent.utils.common import create_enum_class


class TestToolSchema(unittest.TestCase):
    openapi_file = "./tests/fixtures/openapi.yaml"
    examples_file = "./tests/fixtures/examples.yaml"

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

        self.assertEqual(function_call_schemas[0]["name"], "单词本/v1/getWordbook")
        self.assertEqual(function_call_schemas[0]["responses"]["required"], ["wordbook"])
        self.assertEqual(function_call_schemas[0]["responses"]["properties"]["wordbook"]["type"], "array")
        self.assertEqual(function_call_schemas[3]["name"], "单词本/v1/deleteWord")

    def test_function_call_schemas_with_examples(self):
        toolkit = RemoteToolkit.from_openapi_file(self.openapi_file)
        toolkit.examples = toolkit.load_examples_yaml(self.examples_file)
        function_call_schemas = [tool.function_call_schema() for tool in toolkit.get_tools()]
        self.assertEqual(len(function_call_schemas), 4)
        self.assertEqual(function_call_schemas[0]["name"], "单词本/v1/getWordbook")
        self.assertEqual(
            function_call_schemas[0]["examples"][1]["function_call"]["name"], "单词本/v1/getWordbook"
        )
        self.assertEqual(function_call_schemas[0]["responses"]["required"], ["wordbook"])
        self.assertEqual(function_call_schemas[0]["responses"]["properties"]["wordbook"]["type"], "array")
        self.assertEqual(function_call_schemas[3]["name"], "单词本/v1/deleteWord")
        self.assertEqual(
            function_call_schemas[3]["examples"][1]["function_call"]["name"], "单词本/v1/deleteWord"
        )

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
        toolkit = RemoteToolkit.from_openapi_file(self.openapi_file)
        toolkit.examples = toolkit.load_examples_yaml(self.examples_file)
        self.assertEqual(len(toolkit.examples), 12)

        # add_word examples
        examples = toolkit.get_tool("getWordbook").examples

        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0].content, "展示单词列表")
        # function_call name in examples should have `tool_name_prefix` prepended
        self.assertEqual(examples[1].function_call["name"], "单词本/v1/getWordbook")
        self.assertEqual(examples[1].function_call["thoughts"], "这是一个展示单词本的需求")

    def test_dynamic_enum_class(self):
        # 使用函数创建枚举类
        member_names = ["MEMBER1", "MEMBER2", "MEMBER3"]
        MyEnum = create_enum_class("MyEnum", member_names)

        self.assertTrue(issubclass(MyEnum, Enum))
        self.assertTrue(isclass(MyEnum))
        self.assertEqual(MyEnum.MEMBER1.value, "MEMBER1")
        self.assertListEqual(list(MyEnum.__members__.keys()), member_names)

    def test_prompt_parsing(self):
        expected_openapi_dict = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "测试名称",
                },
            },
            "required": ["name"],
            "x-ebagent-prompt": "abc",
        }
        tool = ToolParameterView.from_openapi_dict(expected_openapi_dict)
        self.assertEqual(tool.__prompt__, "abc")


class TestDataTypeSchema(unittest.IsolatedAsyncioTestCase):
    def test_enum_file(self):
        file = "./tests/fixtures/tools/enum_openapi.yaml"
        toolkit = RemoteToolkit.from_openapi_file(file)

        # get_tool
        tool = toolkit.get_tool("OCR")
        self.assertIsNotNone(tool)

        # test function_call schema
        function_call_schema = tool.function_call_schema()

        self.assertIn("language_type", function_call_schema["parameters"]["properties"])
        self.assertIn("enum", function_call_schema["parameters"]["properties"]["language_type"])
        self.assertEqual(len(function_call_schema["parameters"]["properties"]["language_type"]["enum"]), 25)

    def test_empty_object(self):
        file = "./tests/fixtures/tools/empty_object_openapi.yaml"
        toolkit = RemoteToolkit.from_openapi_file(file)
        tool = toolkit.get_tool("OCR")
        function_call_schema = tool.function_call_schema()
        self.assertEqual(function_call_schema["parameters"]["properties"]["language_type"]["type"], "object")


class TestResponseContainsFile(unittest.TestCase):
    def _test_file(self, filename):
        self.assertTrue(tool_response_contains_file({"response": filename}))
        self.assertFalse(tool_response_contains_file({"response": "12345"}))

    def _test_nested_file(self, filename):
        self.assertTrue(tool_response_contains_file({"response": {"file_id": [filename]}}))
        self.assertTrue(tool_response_contains_file({"response": {"file_id": filename}}))
        self.assertTrue(tool_response_contains_file({"response": [filename]}))
        self.assertTrue(tool_response_contains_file({"response": {"file_id": filename, "test": 1}}))
        self.assertFalse(tool_response_contains_file({"response": {"test": 1}}))
        self.assertFalse(tool_response_contains_file({"response": {"test": 1, "test_list": ["123"]}}))
        self.assertFalse(tool_response_contains_file({"response": ["123"]}))

    def test_local_file(self):
        self._test_file(filename="file-local-609f02c0-98c3-11ee-a72b-fa2020087eb4")
        self._test_nested_file(filename="file-local-609f02c0-98c3-11ee-a72b-fa2020087eb4")

    def test_remote_file(self):
        self._test_file(filename="file-123456789012345")
        self._test_nested_file(filename="file-123456789012345")
