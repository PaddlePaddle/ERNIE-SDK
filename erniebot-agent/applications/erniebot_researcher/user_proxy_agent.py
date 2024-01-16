from typing import Optional

from erniebot_agent.memory import SystemMessage


class UserProxyAgent:
    # refer to https://github.com/microsoft/autogen/blob/main/autogen/agentchat/user_proxy_agent.py
    DEFAULT_SYSTEM_MESSAGE = """你是一个有用的人工助手"""

    def __init__(
        self,
        name: str,
        system_message: Optional[SystemMessage] = None,
        human_input_mode: Optional[str] = "ALWAYS",
    ):
        self.name = name
        self.system_message = (
            system_message.content if system_message is not None else self.DEFAULT_SYSTEM_MESSAGE
        )
        self.human_input_mode = human_input_mode

    async def _async_run(self, draft):
        if self.human_input_mode == "ALWAYS":
            return self.get_human_input(draft)
        # TDOD: implement other modes

    def get_human_input(self, prompt: str) -> str:
        """Get human input.

        Override this method to customize the way to get human input.

        Args:
            prompt (str): prompt for the human input.

        Returns:
            str: human input.
        """
        reply = input(prompt + "\n")
        return reply
