# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

from typing import Any, List, Optional

from erniebot_agent.messages import AIMessage, Message


class MockErnieBot:
    """Looks like a LLM but only return what you say. For test only"""

    def __init__(
        self,
        model: str,
        api_type: None,
        access_token: None,
    ):
        self.model = model
        self.api_type = api_type
        self.access_token = access_token

    async def async_chat(
        self,
        messages: List[Message],
        stream: Optional[bool] = False,
        functions: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> Message:
        return AIMessage(messages[0].content)
