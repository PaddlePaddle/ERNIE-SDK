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

from typing import Any, AsyncIterator, Dict, List, Literal, Optional, Union, overload

from erniebot_agent.chat_models.base import ChatModel
from erniebot_agent.messages import Message, response_to_message

import erniebot


class ERNIEBot(ChatModel):
    def __init__(
        self, model: str, api_type: Optional[str] = None, access_token: Optional[str] = None
    ) -> None:
        """Initializes an instance of the `ERNIEBot` class.

        Args:
            model (str): The model name. It should be "ernie-bot", "ernie-bot-turbo", "ernie-bot-8k", or
                "ernie-bot-4".
            api_type (Optional[str]): The API type for erniebot. It should be "aistudio" or "qianfan".
            access_token (Optional[str]): The access token for erniebot.
        """
        super().__init__(model=model)
        self.api_type = api_type
        self.access_token = access_token

    @overload
    async def run(
        self,
        messages: List[Message],
        *,
        stream: Literal[False] = ...,
        functions: Optional[List[dict]] = ...,
        **kwargs: Any,
    ) -> Message:
        ...

    @overload
    async def run(
        self,
        messages: List[Message],
        *,
        stream: Literal[True],
        functions: Optional[List[dict]] = ...,
        **kwargs: Any,
    ) -> AsyncIterator[Message]:
        ...

    @overload
    async def run(
        self, messages: List[Message], *, stream: bool, functions: Optional[List[dict]] = ..., **kwargs: Any
    ) -> Union[Message, AsyncIterator[Message]]:
        ...

    async def run(
        self,
        messages: List[Message],
        *,
        stream: bool = False,
        functions: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> Union[Message, AsyncIterator[Message]]:
        """Asynchronously chats with the ERNIE Bot model.

        Args:
            messages (List[Message]): A list of messages.
            stream (bool): Whether to use streaming generation. Defaults to False.
            functions (Optional[List[dict]]): The function definitions to be used by the model. Defaults to None.
            **kwargs: Keyword arguments, such as `top_p`, `temperature`, `penalty_score`, and `system`.

        Returns:
            If `stream` is False, returns a single message.
            If `stream` is True, returns an asynchronous iterator of messages.
        """
        cfg_dict: Dict[str, Any] = {"model": self.model, "_config_": {}}
        if self.api_type is not None:
            cfg_dict["_config_"]["api_type"] = self.api_type
        if self.access_token is not None:
            cfg_dict["_config_"]["access_token"] = self.access_token

        # TODO: process system message
        cfg_dict["messages"] = [m.to_dict() for m in messages]
        if functions is not None:
            cfg_dict["functions"] = functions

        name_list = ["top_p", "temperature", "penalty_score", "system"]
        for name in name_list:
            if name in kwargs:
                cfg_dict[name] = kwargs[name]

        if stream:
            response = await erniebot.ChatCompletion.acreate(stream=True, **cfg_dict)
            return (response_to_message(d) async for d in response)
        else:
            response = await erniebot.ChatCompletion.acreate(stream=False, **cfg_dict)
            return response_to_message(response)
