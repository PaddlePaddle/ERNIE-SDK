from .conversable_agent import ConversableAgent
from typing import Callable, Dict, Optional


class AssistantAgent(ConversableAgent):
    DEFAULT_SYSTEM_MESSAGE = """您是一位有用的人工智能助手。请利用您的语言技能解决任务。
    如果需要，请逐步解决任务。如果未提供计划，请先解释您的计划。
    当你找到答案时，请仔细验证答案。
    如果可能，请在您的回复中包含可验证的证据。
    当一切完成后，最后回复'终止'
    """

    def __init__(
        self,
        name: str,
        llm_config: Dict,
        system_message: str = DEFAULT_SYSTEM_MESSAGE,
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Optional[str] = "NEVER",
        **kwargs,
    ):
        super().__init__(
            name,
            llm_config,
            system_message,
            is_termination_msg,
            max_consecutive_auto_reply,
            human_input_mode,
            **kwargs,
        )
