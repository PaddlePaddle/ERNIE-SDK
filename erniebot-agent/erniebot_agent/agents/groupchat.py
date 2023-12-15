import logging
import sys
import random
from dataclasses import dataclass
from typing import Dict, List, Optional
import re
from .agentchat import AgentChat
from .conversable_agent import ConversableAgent

logger = logging.getLogger(__name__)
import copy


@dataclass
class GroupChat:
    agents: List[ConversableAgent]
    messages: List[Dict]
    max_round: int = 10
    admin_name: str = "Admin"
    func_call_filter: bool = True
    speaker_selection_method: str = "auto"
    allow_repeat_speaker: bool = True

    _VALID_SPEAKER_SELECTION_METHODS = ["auto", "manual", "random", "round_robin"]

    @property
    def agent_names(self) -> List[str]:
        """Return the names of the agents in the group chat."""
        return [agent.name for agent in self.agents]

    def reset(self):
        """Reset the group chat."""
        self.messages.clear()

    def agent_by_name(self, name):
        """Returns the agent with a given name."""
        return self.agents[self.agent_names.index(name)]

    def next_agent(self, agent: AgentChat, agents: List[ConversableAgent]):
        """Return the next agent in the list."""
        if agents == self.agents:
            return agents[(self.agent_names.index(agent.name) + 1) % len(agents)]
        else:
            offset = self.agent_names.index(agent.name) + 1
            for i in range(len(self.agents)):
                if self.agents[(offset + i) % len(self.agents)] in agents:
                    return self.agents[(offset + i) % len(self.agents)]

    def select_speaker_msg(self, agents: List[ConversableAgent]):
        """Return the message for selecting the next speaker."""
        return f"""您正在玩角色扮演游戏。可以使用以下角色：
        {self._participant_roles(agents)}.
        阅读下面的对话。
        然后从 {[agent.name for agent in agents]}选择下一个角色去扮演。只返回角色。"""

    def manual_select_speaker(self, agents: List[ConversableAgent]):
        """Manually select the next speaker."""

        print("请从以下列表中选择下一位agent")
        _n_agents = len(agents)
        for i in range(_n_agents):
            print(f"{i+1}: {agents[i].name}")
        try_count = 0
        # Assume the user will enter a valid number within 3 tries,
        # otherwise use auto selection to avoid blocking.
        while try_count <= 3:
            try_count += 1
            if try_count >= 3:
                print(f"你已经尝试{try_count} 次了。下一个agent将自动选择。")
                break
            try:
                strs = input("请输入你想要选择agent的序号。当输入q或者不进行输入时则自动选择下一个agent。 ")
                if strs == "" or strs == "q":
                    break
                nums = int(strs)
                if nums > 0 and nums <= _n_agents:
                    return agents[nums - 1]
                else:
                    raise ValueError
            except ValueError:
                print(f"无效输入。请输入1到{_n_agents}的数字。")
        return None

    def select_speaker(self, last_speaker: ConversableAgent, selector: ConversableAgent):
        """Select the next speaker."""
        if self.speaker_selection_method.lower() not in self._VALID_SPEAKER_SELECTION_METHODS:
            raise ValueError(
                f"GroupChat speaker_selection_method is set to '{self.speaker_selection_method}'. "
                f"It should be one of {self._VALID_SPEAKER_SELECTION_METHODS} (case insensitive). "
            )

        agents = self.agents
        n_agents = len(agents)
        # Warn if GroupChat is underpopulated
        if n_agents < 2:
            raise ValueError(
                f"GroupChat is underpopulated with {n_agents} agents. "
                "Please add more agents to the GroupChat or use direct communication instead."
            )
        elif (
            n_agents == 2
            and self.speaker_selection_method.lower() != "round_robin"
            and self.allow_repeat_speaker
        ):
            logger.warning(
                f"GroupChat is underpopulated with {n_agents} agents. "
                + "It is recommended to set speaker_selection_method to "
                + "'round_robin' or allow_repeat_speaker to False."
                + "Or, use direct communication instead."
            )

        if self.func_call_filter and self.messages and "function_call" in self.messages[-1]:
            # find agents with the right function_map which contains the function name
            agents = [
                agent
                for agent in self.agents
                if agent.can_execute_function(self.messages[-1]["function_call"]["name"])
            ]
            if len(agents) == 1:
                # only one agent can execute the function
                return agents[0]
            elif not agents:
                # find all the agents with function_map
                agents = [agent for agent in self.agents if agent.function_map]
                if len(agents) == 1:
                    return agents[0]
                elif not agents:
                    raise ValueError(
                        f"No agent can execute the function {self.messages[-1]['name']}. "
                        "Please check the function_map of the agents."
                    )

        # remove the last speaker from the list to avoid selecting
        # the same speaker if allow_repeat_speaker is False
        agents = (
            agents if self.allow_repeat_speaker else [agent for agent in agents if agent != last_speaker]
        )
        if self.speaker_selection_method.lower() == "manual":
            selected_agent = self.manual_select_speaker(agents)
            if selected_agent:
                return selected_agent
        elif self.speaker_selection_method.lower() == "round_robin":
            return self.next_agent(last_speaker, agents)
        elif self.speaker_selection_method.lower() == "random":
            return random.choice(agents)
        selector.update_system_message(self.select_speaker_msg(agents))
        messages = copy.deepcopy(self.messages)
        final, name = selector.generate_oai_reply(
            messages,
            systems=f"读一下上面的对话。然后从{[agent.name for agent in agents]}选择下一个扮演的角色。仅仅返回角色。",
        )
        if not final:
            # the LLM client is None, thus no reply is generated. Use round robin instead.
            return self.next_agent(last_speaker, agents)

        # If exactly one agent is mentioned, use it. Otherwise, leave the OAI response unmodified

        mentions = self._mentioned_agents(name, agents)
        if len(mentions) == 1:
            name = next(iter(mentions))
        else:
            logger.warning(
                "GroupChat select_speaker failed to resolve "
                + "the next speaker's name. This is because the "
                + f"speaker selection OAI call returned:\n{name}"
            )

        # Return the result
        try:
            return self.agent_by_name(name)
        except ValueError:
            return self.next_agent(last_speaker, agents)

    def _participant_roles(self, agents):
        # Default to all agents registered
        if agents:
            agents = self.agents

        roles = []
        for agent in agents:
            if agent.system_message.strip() == "":
                logger.warning(
                    f"The agent '{agent.name}' has an empty system_message, "
                    + "and may not work well with GroupChat."
                )
            roles.append(f"{agent.name}: {agent.system_message}")
        return "\n".join(roles)

    def _mentioned_agents(self, message_content, agents: List[ConversableAgent]):
        """
        Finds and counts agent mentions in the string message_content,
        taking word boundaries into account.

        Returns: A dictionary mapping agent names to mention counts
        (to be included, at least one mention must occur)
        """
        mentions = dict()
        for agent in agents:
            regex = (
                r"(?<=\W)" + re.escape(agent.name) + r"(?=\W)"
            )  # Finds agent mentions, taking word boundaries into account
            count = len(
                re.findall(regex, " " + message_content + " ")
            )  # Pad the message to help with matching
            if count > 0:
                mentions[agent.name] = count
        return mentions


class GroupChatManager(ConversableAgent):
    """(In preview) A chat manager agent that can manage a group chat of multiple agents."""

    def __init__(
        self,
        groupchat: GroupChat,
        name: str = "chat_manager",
        # unlimited consecutive auto reply by default
        max_consecutive_auto_reply: Optional[int] = sys.maxsize,
        human_input_mode: Optional[str] = "NEVER",
        system_message: str = "Group chat manager.",
        **kwargs,
    ):
        super().__init__(
            name=name,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode=human_input_mode,
            system_message=system_message,
            **kwargs,
        )
        # Order of register_reply is important.
        # Allow sync chat if initiated using initiate_chat
        self.register_reply(
            AgentChat, GroupChatManager.run_chat, config=groupchat, reset_config=GroupChat.reset
        )
        # Allow async chat if initiated using a_initiate_chat
        self.register_reply(
            AgentChat, GroupChatManager.a_run_chat, config=groupchat, reset_config=GroupChat.reset
        )

    def run_chat(
        self,
        sender: ConversableAgent,
        config: GroupChat,
        messages: Optional[List[Dict]] = None,
    ):
        """Run a group chat."""
        if messages is None:
            messages = self._oai_messages[sender]
        message = messages[-1]
        speaker = sender
        groupchat = config
        for i in range(groupchat.max_round):
            # set the name to speaker's name if the role is not function
            if message["role"] != "function":
                message["name"] = speaker.name
            groupchat.messages.append(message)
            # broadcast the message to all agents except the speaker
            for agent in groupchat.agents:
                if agent != speaker:
                    self.send(message=message, recipient=agent, request_reply=False, silent=True)
            if i == groupchat.max_round - 1:
                # the last round
                break
            try:
                # select the next speaker
                speaker = groupchat.select_speaker(speaker, self)
                # let the speaker speak
                reply = speaker.generate_reply(sender=self)
            except KeyboardInterrupt:
                # let the admin agent speak if interrupted
                if groupchat.admin_name in groupchat.agent_names:
                    # admin agent is one of the participants
                    speaker = groupchat.agent_by_name(groupchat.admin_name)
                    reply = speaker.generate_reply(sender=self)
                else:
                    # admin agent is not found in the participants
                    raise
            if reply is None:
                break
            # The speaker sends the message without requesting a reply
            speaker.send(message=reply, recipient=self, request_reply=False)
            update_message = self.last_message(speaker)
            if update_message:
                message = update_message
            else:
                break
        return True, None

    async def a_run_chat(
        self,
        sender: ConversableAgent,
        config: GroupChat,
        messages: Optional[List[Dict]] = None,
    ):
        """Run a group chat asynchronously."""
        if messages is None:
            messages = self._oai_messages[sender]
        message = messages[-1]
        speaker = sender
        groupchat = config
        for i in range(groupchat.max_round):
            # set the name to speaker's name if the role is not function
            if message["role"] != "function":
                message["name"] = speaker.name
            groupchat.messages.append(message)
            # broadcast the message to all agents except the speaker
            for agent in groupchat.agents:
                if agent != speaker:
                    await self.a_send(message=message, recipient=agent, request_reply=False, silent=True)
            if i == groupchat.max_round - 1:
                # the last round
                break
            try:
                # select the next speaker
                speaker = groupchat.select_speaker(speaker, self)
                # let the speaker speak
                reply = await speaker.a_generate_reply(sender=self)
            except KeyboardInterrupt:
                # let the admin agent speak if interrupted
                if groupchat.admin_name in groupchat.agent_names:
                    # admin agent is one of the participants
                    speaker = groupchat.agent_by_name(groupchat.admin_name)
                    reply = await speaker.a_generate_reply(sender=self)
                else:
                    # admin agent is not found in the participants
                    raise
            if reply is None:
                break
            # The speaker sends the message without requesting a reply
            await speaker.a_send(message=reply, recipient=self, request_reply=False)
            update_message = self.last_message(speaker)
            if update_message:
                message = update_message
            else:
                break
        return True, None
