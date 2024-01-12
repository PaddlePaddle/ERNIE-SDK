import logging
import random
import sys
from typing import Any, Dict, List, Optional, Union

from editor_actor_agent import EditorActorAgent
from polish_agent import PolishAgent
from ranking_agent import RankingAgent
from research_agent import ResearchAgent
from reviser_actor_agent import ReviserActorAgent
from tools.utils import JsonUtil

from erniebot_agent.chat_models.erniebot import BaseERNIEBot
from erniebot_agent.memory import AIMessage, HumanMessage

logger = logging.getLogger(__name__)
_VALID_SPEAKER_SELECTION_METHODS = ["auto", "manual", "random", "round_robin"]


class GroupChat(JsonUtil):
    def __init__(
        self,
        agents: Any,
        llm: BaseERNIEBot,
        llm_long: BaseERNIEBot,
        max_round: int = 10,
        admin_name: str = "Admin",
        func_call_filter: bool = True,
        speaker_selection_method: str = "auto",
        allow_repeat_speaker: bool = True,
    ):
        self.agents = agents
        self.max_round = max_round
        self.llm = llm
        self.llm_long = llm_long
        self.admin_name = admin_name
        self.func_call_filter = func_call_filter
        self.speaker_selection_method = speaker_selection_method
        self.allow_repeat_speaker = allow_repeat_speaker

    @property
    def agent_names(self) -> List[str]:
        """Return the names of the agents in the group chat."""
        return [agent.name for agent in self.agents]

    def agent_by_name(self, name: str):
        """Returns the agent with a given name."""
        return self.agents[self.agent_names.index(name)]

    def next_agent(self, agent, agents):
        """Return the next agent in the list."""
        idx = self.agent_names.index(agent.name) if agent.name in self.agent_names else -1
        # Return the next agent
        if agents == self.agents:
            return agents[(idx + 1) % len(agents)]
        else:
            offset = idx + 1
            for i in range(len(self.agents)):
                if self.agents[(offset + i) % len(self.agents)] in agents:
                    return self.agents[(offset + i) % len(self.agents)]

    def select_speaker_msg(self, agents) -> str:
        return f"""您正在玩角色扮演游戏。可以使用以下角色：
{self._participant_roles(agents)}.

阅读下面的对话。
从 {[agent.name for agent in agents]}中选择下一个角色来扮演。仅返回扮演的角色。"""

    def select_speaker_prompt(self, agents) -> str:
        strs = ""
        for i in agents:
            strs += i.name + ":" + i.system_message + "\n"
        return f"阅读下面的对话。 从{[agent.name for agent in agents]} 中选择下一个角色来扮演。仅返回扮演的角色。" + strs

    def manual_select_speaker(self, agents):
        logger.info("请从以下列表中选择下一位Agent：")
        _n_agents = len(agents)
        for i in range(_n_agents):
            logger.info(f"{i+1}: {agents[i].name}")
        try_count = 0
        # Assume the user will enter a valid number within 3 tries,
        # otherwise use auto selection to avoid blocking.
        while try_count <= 3:
            try_count += 1
            if try_count >= 3:
                logger.info(f"你已经尝试{try_count}次了。下一个agent将自动选择。")
                break
            try:
                nums = input("输入下一个发言者的序号（不输入任何内容或输入“q”以使用自动选择）：")
                if nums == "" or nums == "q":
                    break
                index = int(nums)
                if index > 0 and index <= _n_agents:
                    return agents[index - 1]
                else:
                    raise ValueError
            except ValueError:
                logger.info(f"输入无效。请输入 1 到 {_n_agents} 之间的数字。")
        return None

    def _prepare_and_select_agents(self, last_speaker):
        if self.speaker_selection_method.lower() not in _VALID_SPEAKER_SELECTION_METHODS:
            raise ValueError(
                f"GroupChat speaker_selection_method is set to '{self.speaker_selection_method}'. "
                f"It should be one of {_VALID_SPEAKER_SELECTION_METHODS} (case insensitive). "
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
                "It is recommended to set speaker_selection_method to 'round_robin' "
                "or allow_repeat_speaker to False."
                "Or, use direct communication instead."
            )
        # remove the last speaker from the list to avoid selecting
        # the same speaker if allow_repeat_speaker is False
        agents = (
            agents if self.allow_repeat_speaker else [agent for agent in agents if agent != last_speaker]
        )

        if self.speaker_selection_method.lower() == "manual":
            selected_agent = self.manual_select_speaker(agents)
        elif self.speaker_selection_method.lower() == "round_robin":
            selected_agent = self.next_agent(last_speaker, agents)
        elif self.speaker_selection_method.lower() == "random":
            selected_agent = random.choice(agents)
        else:
            selected_agent = None
        return selected_agent, agents

    async def select_speaker(self, last_speaker, messages: List):
        """Select the next speaker."""
        selected_agent, agents = self._prepare_and_select_agents(last_speaker)
        if selected_agent:
            return selected_agent
        # auto speaker selection
        respose = await self.llm_long.chat(messages=messages, system=self.select_speaker_prompt(agents))
        if not respose:
            return self.next_agent(last_speaker, agents)
        # If exactly one agent is mentioned, use it. Otherwise, leave the OAI response unmodified
        # Return the result
        try:
            mentions = self._mentioned_agents(respose.content, agents)
            return self.agent_by_name(mentions)
        except Exception as e:
            logger.error(e)
            logger.warning(
                "GroupChat select_speaker failed to resolve the next speaker's name. "
                + f"This is because the speaker selection OAI call returned:\n{respose}"
            )
            return self.next_agent(last_speaker, agents)

    def _participant_roles(self, agents) -> str:
        # Default to all agents registered
        if agents is None:
            agents = self.agents

        roles = []
        for agent in agents:
            if agent.system_message.strip() == "":
                logger.warning(
                    f"The agent '{agent.system_message}' has an empty description, "
                    + "and may not work well with GroupChat."
                )
            roles.append(f"{agent.name}: {agent.system_message}".strip())
        return "\n".join(roles)

    def _mentioned_agents(self, message_content: str, agents) -> str:
        # Cast message content to str
        next_agent_name = self.parse_json(message_content)["next_agent_name"]
        return next_agent_name


class GroupChatManager:
    """(In preview) A chat manager agent that can manage a group chat of multiple agents."""

    def __init__(
        self,
        groupchat: GroupChat,
        name: Optional[str] = "chat_manager",
        # unlimited consecutive auto reply by default
        max_consecutive_auto_reply: Optional[int] = sys.maxsize,
        human_input_mode: bool = True,
        system_message: Optional[Union[str, List]] = "Group chat manager.",
        **kwargs,
    ):
        self.groupchat = groupchat
        self.name = name
        self.max_consecutive_auto_reply = max_consecutive_auto_reply
        self.human_input_mode = human_input_mode
        self.system_message = system_message

    async def run(
        self,
        query: str,
        report: Union[Dict, str],
        speaker: Any,
    ):
        """Run a group chat."""
        report_list: List[Union[Dict, str]] = [report]
        content = """你需要对生成的报告进行质量检测，请调用已有的各种助手完成这个任务,每次只调用1个助手。
            现在已经存在一份报告，你必须对它进行质量检测，检测后，如果你认为报告质量没有达到要求，你可以调用报告生成助手，重新生成报告。
            请你需要返回一个json格式的字符串,{"next_agent_name":"下一次调用助手的名字"}"""
        messages: List[Union[HumanMessage, AIMessage]] = [HumanMessage(content)]
        notes = ""
        for i in range(self.groupchat.max_round):
            if i == self.groupchat.max_round - 1:
                # the last round
                break
            try:
                # select the next speaker
                speaker = await self.groupchat.select_speaker(speaker, messages)
                # if speaker
            except KeyboardInterrupt:
                # let the admin agent speak if interrupted
                if self.groupchat.admin_name in self.groupchat.agent_names:
                    # admin agent is one of the participants
                    speaker = self.groupchat.agent_by_name(self.groupchat.admin_name)
                else:
                    # admin agent is not found in the participants
                    raise
            if isinstance(speaker, EditorActorAgent):
                respose = await speaker.run(report)
                notes = respose.get("notes", "")
                messages.append(AIMessage("调用" + speaker.name + "得到的结果为" + str(respose)))
            elif isinstance(speaker, ReviserActorAgent):
                report_list.append(await speaker.run(report, notes))
                report = report_list[-1]
                messages.append(AIMessage("调用" + speaker.name + "对报告进行了修订"))
            elif isinstance(speaker, ResearchAgent):
                report_str, paragraphs = await speaker.run(query)
                report_list.append({"report": report_str, "paragraphs": paragraphs})
                messages.append(AIMessage("调用" + speaker.name + "重新生成了一份报告"))
            elif isinstance(speaker, RankingAgent):
                report_list, report = await speaker.run(report_list, query)
                messages.append(AIMessage("调用" + speaker.name + "对多个报告进行了排序，得到最优的报告"))
            elif isinstance(speaker, PolishAgent):
                response = await speaker.run(report["report"], report["paragraphs"])  # type: ignore
                report: Dict = {"report": response[0], "paragraphs": report["paragraphs"]}  # type: ignore
                report_list.append(report)

            if self.human_input_mode:
                reply = input("是否停止，如果您认为生成的report符合要求，则请输入yes，否则输入no\n请输入：")
                if reply == "yes":
                    break
            messages.append(HumanMessage(content))
        return report
