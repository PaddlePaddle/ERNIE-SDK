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

from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Literal,
    Optional,
    Type,
    TypeVar,
    Union,
    overload,
)

from erniebot_agent.chat_models.base import ChatModel
from erniebot_agent.messages import AIMessage, AIMessageChunk, FunctionCall, Message

import erniebot
from erniebot.resources.chat_completion import ChatCompletionResponse
from erniebot.response import EBResponse

_T = TypeVar("_T", AIMessage, AIMessageChunk)


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
    async def async_chat(
        self,
        messages: List[Message],
        *,
        stream: Literal[False] = ...,
        functions: Optional[List[dict]] = ...,
        **kwargs: Any,
    ) -> AIMessage:
        ...

    @overload
    async def async_chat(
        self,
        messages: List[Message],
        *,
        stream: Literal[True],
        functions: Optional[List[dict]] = ...,
        **kwargs: Any,
    ) -> AsyncIterator[AIMessageChunk]:
        ...

    @overload
    async def async_chat(
        self, messages: List[Message], *, stream: bool, functions: Optional[List[dict]] = ..., **kwargs: Any
    ) -> Union[AIMessage, AsyncIterator[AIMessageChunk]]:
        ...

    async def async_chat(
        self,
        messages: List[Message],
        *,
        stream: bool = False,
        functions: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> Union[AIMessage, AsyncIterator[AIMessageChunk]]:
        """Asynchronously chats with the ERNIE Bot model.

        Args:
            messages (List[Message]): A list of messages.
            stream (bool): Whether to use streaming generation. Defaults to False.
            functions (Optional[List[dict]]): The function definitions to be used by the model.
                Defaults to None.
            **kwargs: Keyword arguments, such as `top_p`, `temperature`, `penalty_score`, and `system`.

        Returns:
            If `stream` is False, returns a single message.
            If `stream` is True, returns an asynchronous iterator of message chunks.
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

        name_list = ["top_p", "temperature", "penalty_score", "system", "plugins"]
        for name in name_list:
            if name in kwargs:
                cfg_dict[name] = kwargs[name]
        if "plugins" in cfg_dict and (cfg_dict["plugins"] is None or len(cfg_dict["plugins"]) == 0):
            cfg_dict.pop("plugins")

        # TODO: Improve this when erniebot typing issue is fixed.
        if cfg_dict.get("plugins", None):
            response = await erniebot.ChatCompletionWithPlugins.acreate(  # type: ignore
                messages=cfg_dict["messages"],
                plugins=cfg_dict["plugins"],  # type: ignore
                stream=stream,
                _config_=cfg_dict["_config_"],
                functions=functions,  # type: ignore
                extra_params={"extra_data": '{"multi_step_tool_call_close":false}'},
            )
        else:
            response = await erniebot.ChatCompletion.acreate(stream=stream, **cfg_dict)  # type: ignore
        if isinstance(response, EBResponse):
            return self.convert_response_to_output(response, AIMessage)
        else:
            return (
                self.convert_response_to_output(resp, AIMessageChunk)
                async for resp in response  # type: ignore
            )

    @staticmethod
    def convert_response_to_output(response: EBResponse, output_type: Type[_T]) -> _T:
        if hasattr(response, "function_call"):
            function_call = FunctionCall(
                name=response.function_call["name"],
                thoughts=response.function_call["thoughts"],
                arguments=response.function_call["arguments"],
            )
            return output_type(content="", function_call=function_call, token_usage=response.usage)
        elif isinstance(response, ChatCompletionResponse):
            return output_type(content=response.rbody, function_call=None, token_usage=None)
        else:
            return output_type(content=response.result, function_call=None, token_usage=response.usage)
