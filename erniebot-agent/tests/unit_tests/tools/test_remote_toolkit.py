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

import pytest
from openapi_spec_validator.readers import read_from_filename
from pydantic import Field

from erniebot_agent.tools import RemoteToolkit
from erniebot_agent.tools.schema import (
    ToolParameterView,
    get_typing_list_type,
    is_optional_type,
    json_type,
)
from erniebot_agent.tools.utils import tool_response_contains_file
from erniebot_agent.tools.remote_tool import check_json_length
from erniebot_agent.utils.common import create_enum_class
from erniebot_agent.utils.exceptions import RemoteToolError
from erniebot_agent.tools.current_time_tool import CurrentTimeTool



def test_check_json_length():
    fake_json_data = {
        "key": "1" * 4097
    }
    with pytest.raises(RemoteToolError):
        check_json_length(fake_json_data)


def test_tool_string_format():
    tool = CurrentTimeTool()
    string = str(tool)
    assert tool.tool_name in string
    assert tool.description in string
