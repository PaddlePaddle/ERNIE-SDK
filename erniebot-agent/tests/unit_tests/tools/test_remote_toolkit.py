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

import pytest

from erniebot_agent.tools.current_time_tool import CurrentTimeTool
from erniebot_agent.tools.remote_tool import check_json_length
from erniebot_agent.utils.exceptions import RemoteToolError


def test_check_json_length():
    fake_json_data = {"key": "1" * 4097}
    with pytest.raises(RemoteToolError):
        check_json_length(fake_json_data)


def test_tool_string_format():
    tool = CurrentTimeTool()
    string = str(tool)
    assert tool.tool_name in string
    assert tool.description in string
