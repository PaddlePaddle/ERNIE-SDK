from .conversable_agent import ConversableAgent
from typing import Callable, Dict, Optional, Union


class UserProxyAgent(ConversableAgent):
    def __init__(
        self,
        name: str,
        llm_config: Dict,
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Optional[str] = "ALWAYS",
        function_map: Optional[Dict[str, Callable]] = None,
        default_auto_reply: Optional[Union[str, Dict, None]] = "",
        system_message: str = "",
    ):
        super().__init__(
            name,
            llm_config,
            system_message,
            is_termination_msg,
            max_consecutive_auto_reply,
            human_input_mode,
            function_map,
            default_auto_reply,
        )
