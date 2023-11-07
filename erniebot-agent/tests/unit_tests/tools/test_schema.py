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
from typing import List

from erniebot_agent.tools.schema import (
    PluginSchema,
    ToolParameterView,
    get_typing_list_type,
    json_type,
)
from openapi_spec_validator.readers import read_from_filename


class TestToolSchema(unittest.TestCase):
    openapi_file = "./tests/fixtures/openapi.yaml"

    def test_plugin_schema(self):
        schema = PluginSchema.from_openapi_file(self.openapi_file)

        self.assertEqual(schema.info.title, "单词本")
        self.assertEqual(schema.servers[0].url, "http://127.0.0.1:8081")

    def test_load_and_save(self):
        """function_call requires empty fields, eg: items: {}, but yaml file doesn't
        contain any empty field
        """
        spec_dict = read_from_filename(self.openapi_file)
        schema = PluginSchema.from_openapi_file(self.openapi_file)
        saved_spec_dict = schema.to_openapi_dict()
        self.assertEqual(spec_dict[0], saved_spec_dict)

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


# class TestSearchOpenAPIYaml(unittest.TestCase):
#     openapi_file = "./tests/fixtures/openapi-search.yaml"

#     def test_plugin_schema(self):
#         schema = PluginSchema.from_openapi_file(self.openapi_file)

#         self.assertEqual(schema.info.title, "论文检索工具")
#         self.assertEqual(schema.servers[0].url, "http://127.0.0.1:8082")

#     def test_load_and_save(self):
#         spec_dict = read_from_filename(self.openapi_file)
#         schema = PluginSchema.from_openapi_file(self.openapi_file)
#         saved_spec_dict = schema.to_openapi_dict()
#         for key in ["openapi", "info", "servers", "paths", "components"]:
#             a = spec_dict[0][key] == saved_spec_dict[key]
#             print(a)
#         import pdb; pdb.set_trace()
#         self.assertEqual(spec_dict[0], saved_spec_dict)
