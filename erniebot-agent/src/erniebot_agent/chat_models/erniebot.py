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

import json
import logging
from typing import (
    Any,
    AsyncIterator,
    List,
    Literal,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

import erniebot
from erniebot import ChatCompletionResponse

from erniebot_agent.chat_models.base import ChatModel
from erniebot_agent.memory.messages import (
    AIMessage,
    AIMessageChunk,
    FunctionCall,
    Message,
    PluginInfo,
    SearchInfo,
    SystemMessage,
)
from erniebot_agent.utils import config_from_environ as C

_T = TypeVar("_T", AIMessage, AIMessageChunk)

logger = logging.getLogger(__name__)


class BaseERNIEBot(ChatModel):
    @overload
    async def chat(
        self,
        messages: List[Message],
        *,
        stream: Literal[False] = ...,
        functions: Optional[List[dict]] = ...,
        **kwargs: Any,
    ) -> AIMessage:
        ...

    @overload
    async def chat(
        self,
        messages: List[Message],
        *,
        stream: Literal[True],
        functions: Optional[List[dict]] = ...,
        **kwargs: Any,
    ) -> AsyncIterator[AIMessageChunk]:
        ...

    @overload
    async def chat(
        self, messages: List[Message], *, stream: bool, functions: Optional[List[dict]] = ..., **kwargs: Any
    ) -> Union[AIMessage, AsyncIterator[AIMessageChunk]]:
        ...

    async def chat(
        self,
        messages: List[Message],
        *,
        stream: bool = False,
        functions: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> Union[AIMessage, AsyncIterator[AIMessageChunk]]:
        raise NotImplementedError


class ERNIEBot(BaseERNIEBot):
    """The implementation of the ERNIE Bot model.

    Attributes:
        model (str): The model name.
        api_type (str): The backend of the ERNIE Bot model.
        access_token (Optional[str]): The access token corresponding to the backend.
        default_chat_kwargs (Any): A dict for setting default args for chat model,
            the supported keys include `model`, `_config_`, `top_p`, etc.
    """

    def __init__(
        self,
        model: str,
        api_type: str = "aistudio",
        access_token: Optional[str] = None,
        enable_multi_step_tool_call: bool = False,
        **default_chat_kwargs: Any,
    ) -> None:
        """Initializes an instance of the `ERNIEBot` class.

        Args:
            model (str): The model name. It should be "ernie-3.5", "ernie-turbo", "ernie-4.0", or
                "ernie-longtext".
            api_type (str): The backend of erniebot. It should be "aistudio" or "qianfan".
                Defaults to "aistudio".
            access_token (Optional[str]): The access token for the backend of erniebot.
                If access_token is None, the global access_token will be used.
            enable_multi_step_tool_call (bool): Whether to enable the multi-step tool call.
                Defaults to False.
            **default_chat_kwargs: Keyword arguments, such as `_config_`, `top_p`, `temperature`,
                `penalty_score`, and `system`.
        """
        super().__init__(model=model, **default_chat_kwargs)

        self.api_type = api_type
        if access_token is None:
            access_token = C.get_global_access_token()
        self.access_token = access_token
        self._maybe_validate_qianfan_auth()

        self.enable_multi_step_json = json.dumps(
            {"multi_step_tool_call_close": not enable_multi_step_tool_call}
        )

    @overload
    async def chat(
        self,
        messages: List[Message],
        *,
        stream: Literal[False] = ...,
        functions: Optional[List[dict]] = ...,
        **kwargs: Any,
    ) -> AIMessage:
        ...

    @overload
    async def chat(
        self,
        messages: List[Message],
        *,
        stream: Literal[True],
        functions: Optional[List[dict]] = ...,
        **kwargs: Any,
    ) -> AsyncIterator[AIMessageChunk]:
        ...

    @overload
    async def chat(
        self, messages: List[Message], *, stream: bool, functions: Optional[List[dict]] = ..., **kwargs: Any
    ) -> Union[AIMessage, AsyncIterator[AIMessageChunk]]:
        ...

    async def chat(
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
        cfg_dict = self.default_chat_kwargs.copy()
        cfg_dict["model"] = self.model
        cfg_dict.setdefault("_config_", {})

        if self.api_type is not None:
            cfg_dict["_config_"]["api_type"] = self.api_type
        if self.access_token is not None:
            cfg_dict["_config_"]["access_token"] = self.access_token
        if hasattr(self, "ak") and hasattr(self, "sk"):
            cfg_dict["_config_"]["ak"] = self.ak
            cfg_dict["_config_"]["sk"] = self.sk

        if any(isinstance(m, SystemMessage) for m in messages):
            raise ValueError(f"The input messages should not contain SystemMessage: {messages}")
        cfg_dict["messages"] = [m.to_dict() for m in messages]
        if functions is not None:
            cfg_dict["functions"] = functions

        name_list = ["top_p", "temperature", "penalty_score", "system", "plugins"]
        for name in name_list:
            if name in kwargs:
                cfg_dict[name] = kwargs[name]

        if "plugins" in cfg_dict and (cfg_dict["plugins"] is None or len(cfg_dict["plugins"]) == 0):
            cfg_dict.pop("plugins")

        response = await self._generate_response(cfg_dict, stream, functions)

        if not stream:
            assert isinstance(response, ChatCompletionResponse)
            return convert_response_to_output(response, AIMessage)
        else:
            assert isinstance(response, AsyncIterator)
            # We have to do type casting here due to the known mypy issue:
            # https://github.com/python/mypy/issues/16590
            return (
                convert_response_to_output(resp, AIMessageChunk)
                async for resp in cast(AsyncIterator[ChatCompletionResponse], response)
            )

    def _maybe_validate_qianfan_auth(self) -> None:
        if self.api_type == "qianfan":
            if self.access_token is None:
                # 默认选择千帆时，如果设置了access_token，这个access_token不是aistudio的
                if "ak" and "sk" not in self.default_chat_kwargs:
                    ak, sk = C.get_global_aksk()
                    if ak is None or sk is None:
                        raise RuntimeError("Please set at least one of ak+sk or access token.")
                    else:
                        self.ak = ak
                        self.sk = sk
                else:
                    self.ak = self.default_chat_kwargs.pop("ak")
                    self.sk = self.default_chat_kwargs.pop("sk")
            else:
                # If set access_token in environment and pass ak and sk in default_chat_kwargs,
                # the access_token in default_chat_kwargs will be used.
                if "ak" and "sk" in self.default_chat_kwargs:
                    self.ak = self.default_chat_kwargs.pop("ak")
                    self.sk = self.default_chat_kwargs.pop("sk")

    async def _generate_response(
        self, cfg_dict: dict, stream: bool, functions: Optional[List[dict]]
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionResponse]]:
        # TODO: Improve this when erniebot typing issue is fixed.
        # Note: If plugins is not None, erniebot will not use Baidu_search.
        if "plugins" in cfg_dict:
            response = await erniebot.ChatCompletionWithPlugins.acreate(
                messages=cfg_dict["messages"],
                plugins=cfg_dict["plugins"],  # type: ignore
                stream=stream,
                _config_=cfg_dict["_config_"],
                functions=functions,  # type: ignore
                extra_params={
                    "extra_data": self.enable_multi_step_json,
                },
            )
        else:
            response = await erniebot.ChatCompletion.acreate(
                stream=stream,
                extra_params={
                    "extra_data": self.enable_multi_step_json,
                },
                **cfg_dict,
            )

        return response


def convert_response_to_output(response: ChatCompletionResponse, output_type: Type[_T]) -> _T:
    if hasattr(response, "function_call"):
        function_call = FunctionCall(
            name=response.function_call["name"],
            thoughts=response.function_call["thoughts"],
            arguments=response.function_call["arguments"],
        )
        return output_type(
            content="",
            function_call=function_call,
            plugin_info=None,
            search_info=None,
            token_usage=response.usage,
        )
    elif hasattr(response, "plugin_info"):
        plugin_info = PluginInfo(
            names=[
                response["plugin_metas"][i]["pluginNameForModel"]
                for i in range(len(response["plugin_metas"]))
            ]
        )

        return output_type(
            content=response.result,
            function_call=None,
            plugin_info=plugin_info,
            search_info=None,
            token_usage=response.usage,
        )
    elif hasattr(response, "search_info") and len(response.search_info.items()) > 0:
        search_info = SearchInfo(
            results=response.search_info["search_results"],
        )
        return output_type(
            content=response.result,
            function_call=None,
            plugin_info=None,
            search_info=search_info,
            token_usage=response.usage,
        )
    else:
        return output_type(
            content=response.result,
            function_call=None,
            plugin_info=None,
            search_info=None,
            token_usage=response.usage,
        )
