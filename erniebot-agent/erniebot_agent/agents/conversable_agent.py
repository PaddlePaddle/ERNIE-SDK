import asyncio
from collections import defaultdict
import copy
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from .agentchat import AgentChat
import erniebot
import time

try:
    from termcolor import colored
except ImportError:

    def colored(x, *args, **kwargs):
        return x


logger = logging.getLogger(__name__)


class ConversableAgent(AgentChat):
    MAX_CONSECUTIVE_AUTO_REPLY = 100  # maximum number of consecutive auto replies (subject to future change)

    def __init__(
        self,
        name: str,
        llm_config: Dict,
        system_message: str = "你是一个有用的人工智能助手。",
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Optional[str] = "NEVER",
        function_map: Optional[Dict[str, Callable]] = None,
        default_auto_reply: Optional[Union[str, Dict, None]] = "",
    ):
        super().__init__(name)
        self._oai_messages: Dict[AgentChat, List] = defaultdict(list)
        self._oai_system_message = system_message
        self._is_termination_msg = (
            is_termination_msg if is_termination_msg is not None else (lambda x: x.get("content") == "终止")
        )

        erniebot.api_type = llm_config["api_type"]  # type: ignore
        erniebot.access_token = llm_config["access_token"]  # type: ignore
        self.model = llm_config.get("model", "ernie-bot-4")
        self.human_input_mode = human_input_mode
        self._max_consecutive_auto_reply = (
            max_consecutive_auto_reply
            if max_consecutive_auto_reply is not None
            else self.MAX_CONSECUTIVE_AUTO_REPLY
        )
        self._consecutive_auto_reply_counter: Dict[AgentChat, int] = defaultdict(int)
        self._max_consecutive_auto_reply_dict: Dict[AgentChat, int] = defaultdict(
            self.max_consecutive_auto_reply
        )
        self._function_map = {} if function_map is None else function_map
        self._default_auto_reply = default_auto_reply
        self._reply_func_list: List = []
        self.reply_at_receive: Dict[AgentChat, bool] = defaultdict(bool)
        self.register_reply([AgentChat, None], ConversableAgent.generate_oai_reply)
        self.register_reply([AgentChat, None], ConversableAgent.generate_function_call_reply)
        self.register_reply([AgentChat, None], ConversableAgent.generate_async_function_call_reply)
        self.register_reply([AgentChat, None], ConversableAgent.check_termination_and_human_reply)

    def register_reply(
        self,
        trigger: Union[Type[AgentChat], str, AgentChat, Callable[[AgentChat], bool], List],
        reply_func: Callable,
        position: int = 0,
        config: Optional[Any] = None,
        reset_config: Optional[Callable] = None,
    ):
        if not isinstance(trigger, (type, str, AgentChat, Callable, list)):  # type: ignore
            raise ValueError("trigger must be a class, a string, an agent, a callable or a list.")
        self._reply_func_list.insert(
            position,
            {
                "trigger": trigger,
                "reply_func": reply_func,
                "config": copy.copy(config),
                "init_config": config,
                "reset_config": reset_config,
            },
        )

    @property
    def system_message(self):
        return self._oai_system_message

    def update_system_message(self, system_message: str):
        self._oai_system_message = system_message

    def update_max_consecutive_auto_reply(self, value: int, sender: Optional[AgentChat] = None):
        if sender is None:
            self._max_consecutive_auto_reply = value
            for k in self._max_consecutive_auto_reply_dict:
                self._max_consecutive_auto_reply_dict[k] = value
        else:
            self._max_consecutive_auto_reply_dict[sender] = value

    def max_consecutive_auto_reply(self, sender: Optional[AgentChat] = None) -> int:
        return (
            self._max_consecutive_auto_reply
            if sender is None
            else self._max_consecutive_auto_reply_dict[sender]
        )

    @property
    def chat_messages(self) -> Dict[AgentChat, List[Dict]]:
        return self._oai_messages

    def last_message(self, agent: Optional[AgentChat] = None) -> Optional[Dict]:
        if agent is None:
            n_conversations = len(self._oai_messages)
            if n_conversations == 0:
                return None
            if n_conversations == 1:
                for conversation in self._oai_messages.values():
                    return conversation[-1]
            raise ValueError(
                "More than one conversation is found. Please specify the sender to get the last message."
            )
        if agent not in self._oai_messages.keys():
            raise KeyError(
                f"The agent '{agent.name}' is not present in any conversation. "
                + "No history available for this agent."
            )
        return self._oai_messages[agent][-1]

    @staticmethod
    def _message_to_dict(message: Union[Dict, str]) -> Dict[str, Any]:
        """Convert a message to a dictionary.

        The message can be a string or a dictionary. The string will
        be put in the "content" field of the new dictionary.
        """
        if isinstance(message, str):
            return {"content": message}
        elif isinstance(message, dict):
            return message
        else:
            return dict(message)

    def _append_oai_message(self, message: Union[Dict, str], role, conversation_id: AgentChat) -> bool:
        message = self._message_to_dict(message)
        # create oai message to be appended to the oai conversation that can be passed to oai directly.
        oai_message = {
            k: message[k] for k in ("content", "function_call", "name", "context") if k in message
        }
        if "content" not in oai_message:
            if "function_call" in oai_message:
                oai_message["content"] = None
            else:
                return False
        oai_message["role"] = "function" if message.get("role") == "function" else role
        if "function_call" in oai_message:
            oai_message["role"] = "assistant"
            oai_message["function_call"] = dict(oai_message["function_call"])
        if len(self._oai_messages[conversation_id]) == 0:
            oai_message["role"] = "user"
        elif len(self._oai_messages[conversation_id]) > 0:
            if self._oai_messages[conversation_id][-1]["role"] == "user":
                oai_message["role"] = "assistant"
            elif self._oai_messages[conversation_id][-1]["role"] == "assistant":
                oai_message["role"] = "user"
        self._oai_messages[conversation_id].append(oai_message)
        return True

    def send(
        self,
        message: Union[Dict, str],
        recipient: AgentChat,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        valid = self._append_oai_message(message, "assistant", recipient)
        if valid:
            recipient.receive(message=message, sender=self, request_reply=request_reply, silent=silent)
        else:
            raise ValueError(
                "Message can't be converted into a valid ChatCompletion message."
                + "Either content or function_call must be provided."
            )

    async def a_send(
        self,
        message: Union[Dict, str],
        recipient: AgentChat,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        valid = self._append_oai_message(message, "assistant", recipient)
        if valid:
            await recipient.a_receive(
                message=message, sender=self, request_reply=request_reply, silent=silent
            )
        else:
            raise ValueError(
                "Message can't be converted into a valid ChatCompletion message. "
                + "Either content or function_call must be provided."
            )

    def _print_received_message(self, message: Union[Dict, str], sender: AgentChat):
        print(colored(sender.name, "yellow"), "(to", f"{self.name}):\n", flush=True)
        message = self._message_to_dict(message)
        if message.get("role") == "function":
            func_print = f"***** Response from calling function \"{message['name']}\" *****"
            print(colored(func_print, "green"), flush=True)
            print(message["content"], flush=True)
            print(colored("*" * len(func_print), "green"), flush=True)
        else:
            content = message.get("content")
            if content is not None:
                print(content, flush=True)
            if "function_call" in message:
                function_call = dict(message["function_call"])
                func_print = (
                    f"***** Suggested function "
                    f"Call: {function_call.get('name', '(No function name found)')} *****"
                )
                print(colored(func_print, "green"), flush=True)
                print(
                    "Arguments: \n",
                    function_call.get("arguments", "(No arguments found)"),
                    flush=True,
                    sep="",
                )
                print(colored("*" * len(func_print), "green"), flush=True)
        print("\n", "-" * 80, flush=True, sep="")

    def _process_received_message(self, message, sender: AgentChat, silent):
        message = self._message_to_dict(message)
        valid = self._append_oai_message(message, "user", sender)
        if not valid:
            raise ValueError(
                "Received message can't be converted into a valid "
                + "ChatCompletion message. Either content or function_call "
                + "must be provided."
            )
        if not silent:
            self._print_received_message(message, sender)

    def receive(  # type: ignore
        self,  # type: ignore
        sender: AgentChat,  # type: ignore
        message: Union[Dict, str, None] = None,  # type: ignore
        request_reply: Optional[bool] = None,  # type: ignore
        silent: Optional[bool] = False,  # type: ignore
    ):  # type: ignore
        self._process_received_message(message=message, sender=sender, silent=silent)
        if request_reply is False or request_reply is None and self.reply_at_receive[sender] is False:
            return
        reply = self.generate_reply(messages=self.chat_messages[sender], sender=sender)
        if reply is not None:
            self.send(message=reply, recipient=sender, silent=silent)

    async def a_receive(  # type: ignore
        self,  # type: ignore
        sender: AgentChat,  # type: ignore
        message: Union[Dict, str, None] = None,  # type: ignore
        request_reply: Optional[bool] = None,  # type: ignore
        silent: Optional[bool] = False,  # type: ignore
    ):  # type: ignore
        self._process_received_message(message, sender, silent)
        if request_reply is False or request_reply is None and self.reply_at_receive[sender] is False:
            return
        reply = await self.a_generate_reply(sender=sender)
        if reply is not None:
            await self.a_send(message=reply, recipient=sender, silent=silent)

    def _prepare_chat(self, recipient, clear_history):
        self.reset_consecutive_auto_reply_counter(recipient)
        recipient.reset_consecutive_auto_reply_counter(self)
        self.reply_at_receive[recipient] = recipient.reply_at_receive[self] = True
        if clear_history:
            self.clear_history(recipient)
            recipient.clear_history(self)

    def initiate_chat(
        self,
        recipient: "ConversableAgent",
        clear_history: Optional[bool] = True,
        silent: Optional[bool] = False,
        **context,
    ):
        self._prepare_chat(recipient, clear_history)
        self.send(message=self.generate_init_message(**context), recipient=recipient, silent=silent)

    async def a_initiate_chat(
        self,
        recipient: AgentChat,
        clear_history: Optional[bool] = True,
        silent: Optional[bool] = False,
        **context,
    ):
        self._prepare_chat(recipient, clear_history)
        await self.a_send(message=self.generate_init_message(**context), recipient=recipient, silent=silent)

    def reset(self):
        self.clear_history()
        self.reset_consecutive_auto_reply_counter()
        self.stop_reply_at_receive()
        for reply_func_tuple in self._reply_func_list:
            if reply_func_tuple["reset_config"] is not None:
                reply_func_tuple["reset_config"](reply_func_tuple["config"])
            else:
                reply_func_tuple["config"] = copy.copy(reply_func_tuple["init_config"])

    def stop_reply_at_receive(self, sender: Optional[AgentChat] = None):
        if sender is None:
            self.reply_at_receive.clear()
        else:
            self.reply_at_receive[sender] = False

    def reset_consecutive_auto_reply_counter(self, sender: Optional[AgentChat] = None):
        if sender is None:
            self._consecutive_auto_reply_counter.clear()
        else:
            self._consecutive_auto_reply_counter[sender] = 0

    def clear_history(self, agent: Optional[AgentChat] = None):
        if agent is None:
            self._oai_messages.clear()
        else:
            self._oai_messages[agent].clear()

    def generate_oai_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[AgentChat] = None,
        **kwargs,
    ) -> Tuple[bool, Union[str, Dict, None]]:
        systems = kwargs.get("systems", "")
        if messages is None:
            messages = self._oai_messages[sender]  # type: ignore
        if len(messages) % 2 == 0:
            if messages[0]["role"] == "assistant":
                return False, None
            else:
                messages.append({"role": "user", "content": "请你思考任务是否完成，如果完成则输出'终止'即可，否则请输出完成任务的流程"})
        try:
            response = erniebot.ChatCompletion.create(  # type: ignore
                model=self.model, messages=messages, system=self._oai_system_message + systems
            )  # type: ignore
            return True, response.get_result()
        except Exception as e:
            print(e)
            time.sleep(10)
            response = erniebot.ChatCompletion.create(  # type: ignore
                model=self.model, messages=messages, system=self._oai_system_message + systems
            )  # type: ignore
            return True, response.get_result()

    def generate_function_call_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[AgentChat] = None,
        config: Optional[Any] = None,
    ):
        """Generate a reply using function call."""
        if config is None:
            config = self
        if messages is None:
            messages = self._oai_messages[sender]  # type: ignore
        message = messages[-1]
        if "function_call" in message:
            _, func_return = self.execute_function(message["function_call"])
            return True, func_return
        return False, None

    async def generate_async_function_call_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[AgentChat] = None,
        config: Optional[Any] = None,
    ):
        """Generate a reply using async function call."""
        if config is None:
            config = self
        if messages is None:
            messages = self._oai_messages[sender]  # type: ignore
        message = messages[-1]
        if "function_call" in message:
            func_call = message["function_call"]
            func_name = func_call.get("name", "")
            func = self._function_map.get(func_name, None)
            if func and asyncio.coroutines.iscoroutinefunction(func):
                _, func_return = await self.a_execute_function(func_call)
                return True, func_return

        return False, None

    def check_termination_and_human_reply(
        self,
        sender: AgentChat,
        messages: Optional[List[Dict]] = None,
        config: Optional[Any] = None,
    ) -> Tuple[bool, Union[str, Dict, None]]:
        """Check if the conversation should be terminated, and if human reply is provided."""
        if config is None:
            config = self
        if messages is None:
            messages = self._oai_messages[sender]
        message = messages[-1]
        reply = ""
        no_human_input_msg = ""
        if self.human_input_mode == "ALWAYS":
            reply = self.get_human_input(
                f"Provide feedback to {sender.name}. Press enter to skip "
                + "and use auto-reply, or type 'exit' to end the conversation: "
            )
            no_human_input_msg = "NO HUMAN INPUT RECEIVED." if not reply else ""
            # if the human input is empty, and the message is a
            # termination message, then we will terminate the conversation
            reply = reply if reply or not self._is_termination_msg(message) else "exit"
        else:
            if self._consecutive_auto_reply_counter[sender] >= self._max_consecutive_auto_reply_dict[sender]:
                if self.human_input_mode == "NEVER":
                    reply = "exit"
                else:
                    terminate = self._is_termination_msg(message)
                    reply = self.get_human_input(
                        f"Please give feedback to {sender.name}. "
                        + "Press enter or type 'exit' to stop the conversation: "
                        if terminate
                        else f"Please give feedback to {sender.name}. "
                        + "Press enter to skip and use auto-reply, "
                        + "or type 'exit' to stop the conversation: "
                    )
                    no_human_input_msg = "NO HUMAN INPUT RECEIVED." if not reply else ""
                    # if the human input is empty, and the message
                    # is a termination message, then we will terminate the conversation
                    reply = reply if reply or not terminate else "exit"
            elif self._is_termination_msg(message):
                if self.human_input_mode == "NEVER":
                    reply = "exit"
                else:
                    reply = self.get_human_input(
                        f"Please give feedback to {sender.name}. "
                        + "Press enter or type 'exit' to stop the conversation: "
                    )
                    no_human_input_msg = "NO HUMAN INPUT RECEIVED." if not reply else ""
                    # if the human input is empty, and the message is
                    # a termination message, then we will terminate
                    # the conversation
                    reply = reply or "exit"

        # print the no_human_input_msg
        if no_human_input_msg:
            print(colored(f"\n>>>>>>>> {no_human_input_msg}", "red"), flush=True)

        # stop the conversation
        if reply == "exit":
            # reset the consecutive_auto_reply_counter
            self._consecutive_auto_reply_counter[sender] = 0
            return True, None

        # send the human reply
        if reply or self._max_consecutive_auto_reply_dict[sender] == 0:
            # reset the consecutive_auto_reply_counter
            self._consecutive_auto_reply_counter[sender] = 0
            return True, reply

        # increment the consecutive_auto_reply_counter
        self._consecutive_auto_reply_counter[sender] += 1
        if self.human_input_mode != "NEVER":
            print(colored("\n>>>>>>>> USING AUTO REPLY...", "red"), flush=True)
        return False, None

    async def a_check_termination_and_human_reply(
        self,
        sender: AgentChat,
        messages: Optional[List[Dict]] = None,
        config: Optional[Any] = None,
    ) -> Tuple[bool, Union[str, Dict, None]]:
        """(async) Check if the conversation should be terminated,
        and if human reply is provided."""
        if config is None:
            config = self
        if messages is None:
            messages = self._oai_messages[sender]
        message = messages[-1]
        reply = ""
        no_human_input_msg = ""
        if self.human_input_mode == "ALWAYS":
            reply = await self.a_get_human_input(
                f"Provide feedback to {sender.name}. "
                + "Press enter to skip and use auto-reply, "
                + "or type 'exit' to end the conversation: "
            )
            no_human_input_msg = "NO HUMAN INPUT RECEIVED." if not reply else ""
            # if the human input is empty, and the message is a
            # termination message, then we will terminate the conversation
            reply = reply if reply or not self._is_termination_msg(message) else "exit"
        else:
            if self._consecutive_auto_reply_counter[sender] >= self._max_consecutive_auto_reply_dict[sender]:
                if self.human_input_mode == "NEVER":
                    reply = "exit"
                else:
                    terminate = self._is_termination_msg(message)
                    reply = await self.a_get_human_input(
                        f"Please give feedback to {sender.name}. "
                        + "Press enter or type 'exit' to stop the conversation: "
                        if terminate
                        else f"Please give feedback to {sender.name}. "
                        + "Press enter to skip and use auto-reply, or "
                        + "type 'exit' to stop the conversation: "
                    )
                    no_human_input_msg = "NO HUMAN INPUT RECEIVED." if not reply else ""
                    # if the human input is empty, and the message is
                    # a termination message, then we will terminate the conversation
                    reply = reply if reply or not terminate else "exit"
            elif self._is_termination_msg(message):
                if self.human_input_mode == "NEVER":
                    reply = "exit"
                else:
                    reply = await self.a_get_human_input(
                        f"Please give feedback to {sender.name}. "
                        + "Press enter or type 'exit' to stop the conversation: "
                    )
                    no_human_input_msg = "NO HUMAN INPUT RECEIVED." if not reply else ""
                    # if the human input is empty, and the message
                    # is a termination message, then we will
                    # terminate the conversation
                    reply = reply or "exit"

        # print the no_human_input_msg
        if no_human_input_msg:
            print(colored(f"\n>>>>>>>> {no_human_input_msg}", "red"), flush=True)

        # stop the conversation
        if reply == "exit":
            # reset the consecutive_auto_reply_counter
            self._consecutive_auto_reply_counter[sender] = 0
            return True, None

        # send the human reply
        if reply or self._max_consecutive_auto_reply_dict[sender] == 0:
            # reset the consecutive_auto_reply_counter
            self._consecutive_auto_reply_counter[sender] = 0
            return True, reply

        # increment the consecutive_auto_reply_counter
        self._consecutive_auto_reply_counter[sender] += 1
        if self.human_input_mode != "NEVER":
            print(colored("\n>>>>>>>> USING AUTO REPLY...", "red"), flush=True)

        return False, None

    def generate_reply(  # type: ignore
        self,  # type: ignore
        sender: AgentChat,  # type: ignore
        messages: Optional[List[Dict]] = None,  # type: ignore
        exclude: Optional[List[Callable]] = None,  # type: ignore
    ) -> Union[str, Dict, None]:  # type: ignore
        if all((messages is None, sender is None)):
            error_msg = f"Either {messages=} or {sender=} must be provided."
            logger.error(error_msg)
            raise AssertionError(error_msg)

        if messages is None:
            messages = self._oai_messages[sender]
        for reply_func_tuple in self._reply_func_list:
            reply_func = reply_func_tuple["reply_func"]
            if exclude and reply_func in exclude:
                continue
            if asyncio.coroutines.iscoroutinefunction(reply_func):
                continue
            if self._match_trigger(reply_func_tuple["trigger"], sender):
                final, reply = reply_func(
                    self, messages=messages, sender=sender, config=reply_func_tuple["config"]
                )
                if final:
                    return reply
        return self._default_auto_reply

    async def a_generate_reply(  # type: ignore
        self,  # type: ignore
        sender: AgentChat,  # type: ignore
        messages: Optional[List[Dict]] = None,  # type: ignore
        exclude: Optional[List[Callable]] = None,  # type: ignore
    ) -> Union[str, Dict, None]:  # type: ignore
        if all((messages is None, sender is None)):
            error_msg = f"Either {messages=} or {sender=} must be provided."
            logger.error(error_msg)
            raise AssertionError(error_msg)
        if messages is None:
            messages = self._oai_messages[sender]
        for reply_func_tuple in self._reply_func_list:
            reply_func = reply_func_tuple["reply_func"]
            if exclude and reply_func in exclude:
                continue
            if self._match_trigger(reply_func_tuple["trigger"], sender):
                if asyncio.coroutines.iscoroutinefunction(reply_func):
                    final, reply = await reply_func(
                        self, messages=messages, sender=sender, config=reply_func_tuple["config"]
                    )
                else:
                    final, reply = reply_func(
                        self, messages=messages, sender=sender, config=reply_func_tuple["config"]
                    )
                if final:
                    return reply
        return self._default_auto_reply

    def _match_trigger(self, trigger, sender):
        """Check if the sender matches the trigger."""
        if trigger is None:
            return sender is None
        elif isinstance(trigger, str):
            return trigger == sender.name
        elif isinstance(trigger, type):
            return isinstance(sender, trigger)
        elif isinstance(trigger, AgentChat):
            return trigger == sender
        elif isinstance(trigger, Callable):
            return trigger(sender)
        elif isinstance(trigger, list):
            return any(self._match_trigger(t, sender) for t in trigger)
        else:
            raise ValueError(f"Unsupported trigger type: {type(trigger)}")

    def get_human_input(self, prompt: str) -> str:
        reply = input(prompt)
        return reply

    async def a_get_human_input(self, prompt: str) -> str:
        reply = input(prompt)
        return reply

    @staticmethod
    def _format_json_str(jstr):
        result = []
        inside_quotes = False
        last_char = " "
        for char in jstr:
            if last_char != "\\" and char == '"':
                inside_quotes = not inside_quotes
            last_char = char
            if not inside_quotes and char == "\n":
                continue
            if inside_quotes and char == "\n":
                char = "\\n"
            if inside_quotes and char == "\t":
                char = "\\t"
            result.append(char)
        return "".join(result)

    def execute_function(self, func_call):
        func_name = func_call.get("name", "")
        func = self._function_map.get(func_name, None)

        is_exec_success = False
        if func is not None:
            # Extract arguments from a json-like string and put it into a dict.
            input_string = self._format_json_str(func_call.get("arguments", "{}"))
            try:
                arguments = json.loads(input_string)
            except json.JSONDecodeError as e:
                arguments = None
                content = f"Error: {e}\n You argument should follow json format."
            # Try to execute the function
            if arguments is not None:
                print(
                    colored(f"\n>>>>>>>> EXECUTING FUNCTION {func_name}...", "magenta"),
                    flush=True,
                )
                try:
                    content = func(**arguments)
                    is_exec_success = True
                except Exception as e:
                    content = f"Error: {e}"
        else:
            content = f"Error: Function {func_name} not found."
        return is_exec_success, {
            "name": func_name,
            "role": "function",
            "content": str(content),
        }

    async def a_execute_function(self, func_call):
        func_name = func_call.get("name", "")
        func = self._function_map.get(func_name, None)

        is_exec_success = False
        if func is not None:
            # Extract arguments from a json-like string and put it into a dict.
            input_string = self._format_json_str(func_call.get("arguments", "{}"))
            try:
                arguments = json.loads(input_string)
            except json.JSONDecodeError as e:
                arguments = None
                content = f"Error: {e}\n You argument should follow json format."

            # Try to execute the function
            if arguments is not None:
                print(
                    colored(f"\n>>>>>>>> EXECUTING ASYNC FUNCTION {func_name}...", "magenta"),
                    flush=True,
                )
                try:
                    if asyncio.coroutines.iscoroutinefunction(func):
                        content = await func(**arguments)
                    else:
                        # Fallback to sync function if the function is not async
                        content = func(**arguments)
                    is_exec_success = True
                except Exception as e:
                    content = f"Error: {e}"
        else:
            content = f"Error: Function {func_name} not found."

        return is_exec_success, {
            "name": func_name,
            "role": "function",
            "content": str(content),
        }

    def generate_init_message(self, **context) -> Union[str, Dict]:
        return context["message"]

    def register_function(self, function_map: Dict[str, Callable]):
        self._function_map.update(function_map)

    def can_execute_function(self, name: str) -> bool:
        return name in self._function_map

    @property
    def function_map(self) -> Dict[str, Callable]:
        return self._function_map
