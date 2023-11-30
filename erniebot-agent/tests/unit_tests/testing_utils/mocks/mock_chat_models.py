import itertools

from erniebot_agent.chat_models.base import ChatModel
from erniebot_agent.messages import (
    AIMessage,
    FunctionCall,
    FunctionMessage,
    HumanMessage,
)


class FakeSimpleChatModel(ChatModel):
    def __init__(self):
        super().__init__("simple_chat_model")

    @property
    def response(self):
        return AIMessage(content="Text response", function_call=None, token_usage=None)

    async def async_chat(self, messages, *, stream=False, **kwargs):
        if stream:
            raise ValueError("Streaming is not supported.")
        return self.response


class FakeFunctionCallingChatModel(ChatModel):
    def __init__(self):
        super().__init__("function_calling_chat_model")

    @property
    def text_response(self):
        return AIMessage(content="Text response", function_call=None, token_usage=None)

    async def async_chat(self, messages, *, stream=False, functions=None, **kwargs):
        if stream:
            raise ValueError("Streaming is not supported.")
        if functions is None:
            raise TypeError("`functions` must be provided.")
        last_message = messages[-1]
        if isinstance(last_message, HumanMessage):
            fake_function_call = self._create_fake_function_call(functions)
            return AIMessage(content="", function_call=fake_function_call)
        elif isinstance(last_message, FunctionMessage):
            return self.text_response

    def _create_fake_function_call(self, functions):
        function = functions[0]
        parameters = function["parameters"]
        func_name = function["name"]
        func_args = dict(zip(parameters["properties"].keys(), itertools.cycle([None])))
        return FunctionCall(name=func_name, thoughts="Function call response", arguments=func_args)


class FakeChatModelWithPresetResponses(ChatModel):
    def __init__(self, responses):
        super().__init__("chat_model_with_preset_responses")
        self.responses = responses
        self._counter = 0

    async def async_chat(self, messages, *, stream=False, **kwargs):
        if stream:
            raise ValueError("Streaming is not supported.")
        response = self.responses[self._counter]
        self._counter += 1
        return response
