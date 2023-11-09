import itertools

from erniebot_agent.chat_models.base import ChatModel
from erniebot_agent.messages import (
    AIMessage,
    FunctionCall,
    FunctionMessage,
    HumanMessage,
)


class MockSimpleChatModel(ChatModel):
    async def async_chat(self, messages, *, stream=False, **kwargs):
        return AIMessage(content="Text response")


class MockChatModelWithFunctions(ChatModel):
    async def async_chat(self, messages, *, stream=False, functions=None, **kwargs):
        if functions is None:
            raise TypeError("`functions` must be provided.")
        last_message = messages[-1]
        if isinstance(last_message, HumanMessage):
            fake_function_call = self._create_fake_function_call(functions)
            return AIMessage(content="", function_call=fake_function_call)
        elif isinstance(last_message, FunctionMessage):
            return AIMessage(content="Text response")

    def _create_fake_function_call(self, functions):
        function = functions[0]
        parameters = function["parameters"]
        func_name = function["name"]
        func_args = dict(zip(parameters["properties"].keys(), itertools.cycle([None])))
        return FunctionCall(name=func_name, thoughts="Function call response", arguments=func_args)
