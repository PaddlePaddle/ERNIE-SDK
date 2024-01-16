from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Iterator, List, Mapping, Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.chat_models.base import BaseChatModel
from langchain.pydantic_v1 import Field, root_validator
from langchain.schema import ChatGeneration, ChatResult
from langchain.schema.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)
from langchain.schema.output import ChatGenerationChunk
from langchain.utils import get_from_dict_or_env

_MessageDict = Dict[str, Any]


class ErnieBotChat(BaseChatModel):
    """ERNIE Bot Chat large language models API.

    To use, you should have the ``erniebot`` python package installed, and the
    environment variable ``AISTUDIO_ACCESS_TOKEN`` set with your AI Studio access token.

    Example:
        .. code-block:: python
            from erniebot_agent.extensions.langchain.chat_models import ErnieBotChat
            erniebot_chat = ErnieBotChat(model="ernie-3.5")
    """

    client: Any = None
    aistudio_access_token: Optional[str] = None
    """AI Studio access token."""
    max_retries: int = 6
    """Maximum number of retries to make when generating."""
    streaming: bool = False
    """Whether to stream the results or not."""

    model: str = "ernie-3.5"
    """Model to use."""
    temperature: Optional[float] = 0.95
    """Sampling temperature to use."""
    top_p: Optional[float] = 0.8
    """Parameter of nucleus sampling that affects the diversity of generated content."""
    penalty_score: Optional[float] = 1
    """Penalty assigned to tokens that have been generated."""
    request_timeout: Optional[int] = 60
    """How many seconds to wait for the server to send data before giving up."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""

    ernie_client_id: Optional[str] = None
    ernie_client_secret: Optional[str] = None
    """For raising deprecation warnings."""

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling ERNIE Bot API."""
        normal_params = {
            "model": self.model,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "penalty_score": self.penalty_score,
            "request_timeout": self.request_timeout,
        }
        return {**normal_params, **self.model_kwargs}

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return self._default_params

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        """Get the parameters used to invoke the model."""
        auth_cfg: Dict[str, Optional[str]] = {
            "api_type": "aistudio",
            "access_token": self.aistudio_access_token,
        }
        return {**{"_config_": {"max_retries": self.max_retries, **auth_cfg}}, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "erniebot"

    @root_validator()
    def validate_enviroment(cls, values: Dict) -> Dict:
        values["aistudio_access_token"] = get_from_dict_or_env(
            values,
            "aistudio_access_token",
            "AISTUDIO_ACCESS_TOKEN",
        )

        try:
            import erniebot

            values["client"] = erniebot.ChatCompletion
        except ImportError:
            raise ImportError(
                "Could not import erniebot python package. Please install it with `pip install erniebot`."
            )
        return values

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            chunks = self._stream(messages, stop=stop, run_manager=run_manager, **kwargs)
            generation: Optional[ChatGenerationChunk] = None
            for chunk in chunks:
                if generation is None:
                    generation = chunk
                else:
                    generation += chunk
            assert generation is not None
            return ChatResult(generations=[generation])
        else:
            params = self._invocation_params
            params.update(kwargs)
            params["messages"] = self._convert_messages_to_dicts(messages)
            system_prompt = self._build_system_prompt_from_messages(messages)
            if system_prompt is not None:
                params["system"] = system_prompt
            params["stream"] = False
            response = self.client.create(**params)
            return self._build_chat_result_from_response(response)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            chunks = self._astream(messages, stop=stop, run_manager=run_manager, **kwargs)
            generation: Optional[ChatGenerationChunk] = None
            async for chunk in chunks:
                if generation is None:
                    generation = chunk
                else:
                    generation += chunk
            assert generation is not None
            return ChatResult(generations=[generation])
        else:
            params = self._invocation_params
            params.update(kwargs)
            params["messages"] = self._convert_messages_to_dicts(messages)
            system_prompt = self._build_system_prompt_from_messages(messages)
            if system_prompt is not None:
                params["system"] = system_prompt
            params["stream"] = False
            response = await self.client.acreate(**params)
            return self._build_chat_result_from_response(response)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        if stop is not None:
            raise ValueError("Currently, `stop` is not supported when streaming is enabled.")
        params = self._invocation_params
        params.update(kwargs)
        params["messages"] = self._convert_messages_to_dicts(messages)
        system_prompt = self._build_system_prompt_from_messages(messages)
        if system_prompt is not None:
            params["system"] = system_prompt
        params["stream"] = True
        for resp in self.client.create(**params):
            chunk = self._build_chunk_from_response(resp)
            yield chunk
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        if stop is not None:
            raise ValueError("Currently, `stop` is not supported when streaming is enabled.")
        params = self._invocation_params
        params.update(kwargs)
        params["messages"] = self._convert_messages_to_dicts(messages)
        system_prompt = self._build_system_prompt_from_messages(messages)
        if system_prompt is not None:
            params["system"] = system_prompt
        params["stream"] = True
        async for resp in await self.client.acreate(**params):
            chunk = self._build_chunk_from_response(resp)
            yield chunk
            if run_manager:
                await run_manager.on_llm_new_token(chunk.text, chunk=chunk)

    def _build_chat_result_from_response(self, response: Mapping[str, Any]) -> ChatResult:
        message_dict = self._build_dict_from_response(response)
        generation = ChatGeneration(
            message=self._convert_dict_to_message(message_dict),
            generation_info=dict(finish_reason="stop"),
        )
        token_usage = response.get("usage", {})
        llm_output = {"token_usage": token_usage, "model_name": self.model}
        return ChatResult(generations=[generation], llm_output=llm_output)

    def _build_chunk_from_response(self, response: Mapping[str, Any]) -> ChatGenerationChunk:
        message_dict = self._build_dict_from_response(response)
        message = self._convert_dict_to_message(message_dict)
        msg_chunk = AIMessageChunk(
            content=message.content,
            additional_kwargs=message.additional_kwargs,
        )
        return ChatGenerationChunk(message=msg_chunk)

    def _build_dict_from_response(self, response: Mapping[str, Any]) -> _MessageDict:
        message_dict: _MessageDict = {"role": "assistant"}
        if "function_call" in response:
            message_dict["content"] = None
            message_dict["function_call"] = response["function_call"]
        else:
            message_dict["content"] = response["result"]
        return message_dict

    def _build_system_prompt_from_messages(self, messages: List[BaseMessage]) -> Optional[str]:
        system_message_content_list: List[str] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                if isinstance(msg.content, str):
                    system_message_content_list.append(msg.content)
                else:
                    raise TypeError
        if len(system_message_content_list) > 0:
            return "\n".join(system_message_content_list)
        else:
            return None

    def _convert_messages_to_dicts(self, messages: List[BaseMessage]) -> List[dict]:
        erniebot_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                # Ignore system messages, as we handle them elsewhere.
                continue
            eb_msg = self._convert_message_to_dict(msg)
            erniebot_messages.append(eb_msg)
        return erniebot_messages

    @staticmethod
    def _convert_dict_to_message(message_dict: _MessageDict) -> BaseMessage:
        role = message_dict["role"]
        if role == "user":
            return HumanMessage(content=message_dict["content"])
        elif role == "assistant":
            content = message_dict["content"] or ""
            if message_dict.get("function_call"):
                additional_kwargs = {"function_call": dict(message_dict["function_call"])}
            else:
                additional_kwargs = {}
            return AIMessage(content=content, additional_kwargs=additional_kwargs)
        elif role == "function":
            return FunctionMessage(content=message_dict["content"], name=message_dict["name"])
        else:
            return ChatMessage(content=message_dict["content"], role=role)

    @staticmethod
    def _convert_message_to_dict(message: BaseMessage) -> _MessageDict:
        message_dict: _MessageDict
        if isinstance(message, ChatMessage):
            message_dict = {"role": message.role, "content": message.content}
        elif isinstance(message, HumanMessage):
            message_dict = {"role": "user", "content": message.content}
        elif isinstance(message, AIMessage):
            message_dict = {"role": "assistant", "content": message.content}
            if "function_call" in message.additional_kwargs:
                message_dict["function_call"] = message.additional_kwargs["function_call"]
                if message_dict["content"] == "":
                    message_dict["content"] = None
        elif isinstance(message, FunctionMessage):
            message_dict = {
                "role": "function",
                "content": message.content,
                "name": message.name,
            }
        else:
            raise TypeError(f"Got unknown type {message}")

        return message_dict
