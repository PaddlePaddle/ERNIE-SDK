from erniebot.response import EBResponse

from erniebot_agent.chat_models.base import ChatModel
from erniebot_agent.chat_models.erniebot import ERNIEBot
from erniebot_agent.memory import AIMessage


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


class FakeERNIEBotWithPresetResponses(ERNIEBot):
    def __init__(self, fake_response=None):
        super().__init__("ERNIEBot_with_fake_response")
        self.fake_response = fake_response
        if self.fake_response is None:
            raise RuntimeError(
                "You must provide a fake_response to the FakeERNIEBotWithPresetResponses constructor."
            )

    async def _get_response(self, cfg_dict, stream, functions) -> EBResponse:
        return self.fake_response


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
