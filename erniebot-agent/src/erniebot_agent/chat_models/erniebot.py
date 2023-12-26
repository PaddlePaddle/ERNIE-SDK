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
    overload,
)

import erniebot
from erniebot.response import EBResponse

from erniebot_agent.chat_models.base import ChatModel
from erniebot_agent.messages import (
    AIMessage,
    AIMessageChunk,
    FunctionCall,
    Message,
    SearchInfo,
)

_T = TypeVar("_T", AIMessage, AIMessageChunk)

logger = logging.getLogger(__name__)


class ERNIEBot(ChatModel):
    def __init__(
        self,
        model: str,
        api_type: Optional[str] = None,
        access_token: Optional[str] = None,
        enable_multi_step_tool_call: bool = False,
        **default_chat_kwargs: Any,
    ) -> None:
        """Initializes an instance of the `ERNIEBot` class.

        Args:
            model (str): The model name. It should be "ernie-3.5", "ernie-turbo", "ernie-4.0", or
                "ernie-longtext".
            api_type (Optional[str]): The API type for erniebot. It should be "aistudio" or "qianfan".
            access_token (Optional[str]): The access token for erniebot.
            close_multi_step_tool_call (bool): Whether to close the multi-step tool call. Defaults to False.
        """
        super().__init__(model=model, **default_chat_kwargs)

        self.api_type = api_type
        self.access_token = access_token
<<<<<<< HEAD
=======
        self._maybe_validate_qianfan_auth()

>>>>>>> 4e5c710 (Update erniebot.py)
        self.enable_multi_step_json = json.dumps(
            {"multi_step_tool_call_close": not enable_multi_step_tool_call}
        )

<<<<<<< HEAD
=======
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

>>>>>>> 4e5c710 (Update erniebot.py)
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
        cfg_dict = self.default_chat_kwargs.copy()
        cfg_dict["model"] = self.model
        cfg_dict.setdefault("_config_", {})

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

        cfg_dict["extra_params"] = {"extra_data": self.enable_multi_step_json}
        # TODO: Improve this when erniebot typing issue is fixed.
        # Note: If plugins is not None, erniebot will not use Baidu_search.
        if cfg_dict.get("plugins", None):
            # TODO: logger.debug here when cfg_dict is consolidated
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
            cfg_dict["extra_params"] = {"extra_data": self.enable_multi_step_json}
            cfg_dict["stream"] = stream
            logger.debug(f"ERNIEBot Request: {cfg_dict}")
            response = await erniebot.ChatCompletion.acreate(**cfg_dict)
            logger.debug(f"ERNIEBot Response: {response}")
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
            return output_type(
                content="", function_call=function_call, search_info=None, token_usage=response.usage
            )
        elif hasattr(response, "search_info") and len(response.search_info.items()) > 0:
            search_info = SearchInfo(
                results=response.search_info["search_results"],
            )
            return output_type(
                content=response.result,
                function_call=None,
                search_info=search_info,
                token_usage=response.usage,
            )
        else:
            return output_type(
                content=response.result, function_call=None, search_info=None, token_usage=response.usage
            )
