from erniebot_agent.chat_models.base import ChatModel
from erniebot_agent.chat_models.erniebot import ERNIEBot
from erniebot_agent.memory import AIMessage


class FakeSimpleChatModel(ChatModel):
    def __init__(self):
        super().__init__("simple_chat_model")

    @property
    def response(self):
        return AIMessage(content="Text response", function_call=None, token_usage=None)

    async def chat(self, messages, *, stream=False, **kwargs):
        if stream:
            raise ValueError("Streaming is not supported.")
        if "system" in kwargs:
            response = f"Recieved system message: {kwargs['system']}"
            return AIMessage(content=response, function_call=None, token_usage=None)
        return self.response


class FakeERNIEBotWithPresetResponses(ChatModel):
    def __init__(self, responses):
        super().__init__("erniebot_with_preset_responses")
        self.responses = responses
        self._counter = 0

    async def chat(self, messages, *, stream=False, functions=None, **kwargs):
        if stream:
            raise ValueError("Streaming is not supported.")
        # Ignore `messages`, `functions`, and `kwargs`
        response = self.responses[self._counter]
        self._counter += 1
        return response


class FakeERNIEBotWithAllInput(ERNIEBot):
    def __init__(self, model, api_type, access_token, enable_multi_step_tool_call, **default_chat_kwargs):
        super().__init__(model, api_type, access_token, enable_multi_step_tool_call, **default_chat_kwargs)
