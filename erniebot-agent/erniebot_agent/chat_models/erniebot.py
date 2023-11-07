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

from typing import Any, AsyncIterator, Dict, List, Optional, Union

from erniebot_agent.chat_models.base import ChatModel
from erniebot_agent.message import AIMessage, Message

import erniebot


class ERNIEBot(ChatModel):
    def __init__(
        self,
        model: str,
        api_type: Optional[str] = None,
        access_token: Optional[str] = None,
    ):
        """
        Initializes a instance of the ERNIEBot class.
        Args:
            model(str): The model name. It should be ernie-bot, ernie-bot-turbo or ernie-bot-4.
            api_type(Optional[str]): The api-type for ERNIEBot. It should be aistudio or qianfan.
            access_token(Optional[str]): The access token for ERNIEBot.
        """
        super().__init__(model=model)
        self.api_type = api_type
        self.access_token = access_token

    async def async_chat(
        self,
        messages: List[Message],
        stream: Optional[bool] = False,
        functions: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> Union[Message, AsyncIterator[Message]]:
        """
        Asynchronously chat with the LLM.

        Args:
            messages(List[Message]): A list of messages.
            stream(Optional[bool]): Whether to use streaming generation. Defaults to False.
            functions(Optional[List[dict]]): Set the function definitions for the chat model.
                Defaults to None.
            kwargs(Any): Keyword arguments, such as 'top_p', 'temperature', 'penalty_score' and 'system'

        Returns:
            If stream is False, returns a single message.
            If stream is True, returns an asynchronous iterator of messages.
        """
        cfg_dict: Dict[str, Any] = {"model": self.model, "_config_": {}}
        if self.api_type is not None:
            cfg_dict["_config_"]["api_type"] = self.api_type
        if self.access_token is not None:
            cfg_dict["_config_"]["access_token"] = self.access_token

        # TODO: process system message
        cfg_dict["messages"] = [m.to_dict() for m in messages]
        cfg_dict["stream"] = stream
        if functions is not None:
            cfg_dict["functions"] = functions

        name_list = ["top_p", "temperature", "penalty_score", "system"]
        for name in name_list:
            if name in kwargs:
                cfg_dict[name] = kwargs[name]

        response: Any = await erniebot.ChatCompletion.acreate(**cfg_dict)

        if stream:
            return (AIMessage.from_response(d) async for d in response)
        else:
            return AIMessage.from_response(response)
