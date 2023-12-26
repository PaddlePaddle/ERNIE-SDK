import abc
import json
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, final

from erniebot_agent.agents.base import BaseAgent
from erniebot_agent.agents.callback.callback_manager import CallbackManager
from erniebot_agent.agents.callback.default import get_default_callbacks
from erniebot_agent.agents.callback.handlers.base import CallbackHandler
from erniebot_agent.agents.mixins import GradioMixin
from erniebot_agent.agents.schema import (
    AgentResponse,
    LLMResponse,
    ToolResponse,
)
from erniebot_agent.chat_models.base import ChatModel
from erniebot_agent.file import GlobalFileManagerHandler, protocol
from erniebot_agent.file.base import File
from erniebot_agent.file.file_manager import FileManager
from erniebot_agent.file.protocol import extract_file_ids
from erniebot_agent.memory import Memory
from erniebot_agent.memory.messages import Message, SystemMessage
from erniebot_agent.tools.base import BaseTool
from erniebot_agent.tools.tool_manager import ToolManager
from erniebot_agent.utils.exceptions import FileError

_PLUGINS_WO_FILE_IO: Tuple[str] = ("eChart",)


class Agent(GradioMixin, BaseAgent):
    """The base class for agents.

    Typically, this is the class that a custom agent class should inherit from.
    A class inheriting from this class must implement how the agent orchestates
    the components to complete tasks.
    """

    def __init__(
        self,
        llm: ChatModel,
        tools: Union[ToolManager, List[BaseTool]],
        memory: Memory,
        *,
        system_message: Optional[SystemMessage] = None,
        callbacks: Optional[Union[CallbackManager, List[CallbackHandler]]] = None,
        file_manager: Optional[FileManager] = None,
        plugins: Optional[List[str]] = None,
    ) -> None:
        """Initialize an agent.

        Args:
            llm: An LLM for the agent to use.
            tools: A list of tools for the agent to use.
            memory: A memory object that equips the agent to remember chat
                history.
            system_message: A message that tells the LLM how to interpret the
                conversations. If `None`, the system message contained in
                `memory` will be used.
            callbacks: A list of callback handlers for the agent to use. If
                `None`, a default list of callbacks will be used.
            file_manager: A file manager for the agent to interact with files.
                If `None`, a global file manager that can be shared among
                different components will be implicitly created and used.
            plugins: A list of names of the plugins for the agent to use. If
                `None`, the agent will use a default list of plugins. Set
                `plugins` to `[]` to disable the use of plugins.
        """
        super().__init__()
        self._llm = llm
        if isinstance(tools, ToolManager):
            self._tool_manager = tools
        else:
            self._tool_manager = ToolManager(tools)
        self._memory = memory
        if system_message:
            self.system_message = system_message
        else:
            self.system_message = memory.get_system_message()
        if callbacks is None:
            callbacks = get_default_callbacks()
        if isinstance(callbacks, CallbackManager):
            self._callback_manager = callbacks
        else:
            self._callback_manager = CallbackManager(callbacks)
        self._file_manager = file_manager
        self._plugins = plugins
        self._init_file_needs_url()

    @property
    def llm(self) -> ChatModel:
        """The LLM that the agent uses."""
        return self._llm

    @property
    def memory(self) -> Memory:
        """The message storage that keeps the chat history."""
        return self._memory

    @property
    def tools(self) -> List[BaseTool]:
        """The tools that the agent can choose from."""
        return self._tool_manager.get_tools()

    @final
    async def async_run(self, prompt: str, files: Optional[List[File]] = None) -> AgentResponse:
        """Run the agent asynchronously.

        Args:
            prompt: A natural language text describing the task that the agent
                should perform.
            files: The files that the agent can use to perform the task.

        Returns:
            Response from the agent.
        """

        await self._callback_manager.on_run_start(agent=self, prompt=prompt)
        agent_resp = await self._async_run(prompt, files)
        await self._callback_manager.on_run_end(agent=self, response=agent_resp)
        return agent_resp

    def load_tool(self, tool: BaseTool) -> None:
        """Load a tool into the agent.

        Args:
            tool: The tool to load.
        """
        self._tool_manager.add_tool(tool)

    def unload_tool(self, tool: BaseTool) -> None:
        """Unload a tool from the agent.

        Args:
            tool: The tool to unload.
        """
        self._tool_manager.remove_tool(tool)

    def reset_memory(self) -> None:
        """Clear the chat history."""
        self._memory.clear_chat_history()

    @abc.abstractmethod
    async def _async_run(self, prompt: str, files: Optional[List[File]] = None) -> AgentResponse:
        """Run the agent asynchronously.

        This method is the same as `async_run`, except that the callbacks are
        not called.
        """
        raise NotImplementedError

    @final
    async def _async_run_tool(self, tool_name: str, tool_args: str) -> ToolResponse:
        tool = self._tool_manager.get_tool(tool_name)
        await self._callback_manager.on_tool_start(agent=self, tool=tool, input_args=tool_args)
        try:
            tool_resp = await self._async_run_tool_without_hooks(tool, tool_args)
        except (Exception, KeyboardInterrupt) as e:
            await self._callback_manager.on_tool_error(agent=self, tool=tool, error=e)
            raise
        await self._callback_manager.on_tool_end(agent=self, tool=tool, response=tool_resp)
        return tool_resp

    @final
    async def _async_run_llm(self, messages: List[Message], **opts: Any) -> LLMResponse:
        await self._callback_manager.on_llm_start(agent=self, llm=self._llm, messages=messages)
        try:
            llm_resp = await self._async_run_llm_without_hooks(messages, **opts)
        except (Exception, KeyboardInterrupt) as e:
            await self._callback_manager.on_llm_error(agent=self, llm=self._llm, error=e)
            raise
        await self._callback_manager.on_llm_end(agent=self, llm=self._llm, response=llm_resp)
        return llm_resp

    async def _async_run_tool_without_hooks(self, tool: BaseTool, tool_args: str) -> ToolResponse:
        parsed_tool_args = self._parse_tool_args(tool_args)
        # XXX: Sniffing is less efficient and probably unnecessary.
        # Can we make a protocol to statically recognize file inputs and outputs
        # or can we have the tools introspect about this?
        input_files = await self._sniff_and_extract_files_from_args(parsed_tool_args, tool, "input")
        tool_ret = await tool(**parsed_tool_args)
        if isinstance(tool_ret, dict):
            output_files = await self._sniff_and_extract_files_from_args(tool_ret, tool, "output")
        else:
            output_files = []
        tool_ret_json = json.dumps(tool_ret, ensure_ascii=False)
        return ToolResponse(json=tool_ret_json, input_files=input_files, output_files=output_files)

    async def _sniff_and_extract_files_from_args( # TODO(shiyutang): to be tested
        self, args: Dict[str, Any], tool: BaseTool, file_type: Literal["input", "output"]
    ) -> List[File]:
        agent_files: List[File] = []
        for val in args.values():
            if isinstance(val, str):
                file = await self._get_file_from_file_id(val, tool, file_type)
                if file is None:
                    continue
                else:
                    agent_files.append(file)
            elif isinstance(val, dict):
                agent_files.extend(await self._sniff_and_extract_files_from_args(val, tool, file_type))
            elif isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
                for item in val:
                    agent_files.extend(await self._sniff_and_extract_files_from_args(item, tool, file_type))
        return agent_files

    async def _async_run_llm_without_hooks(
        self, messages: List[Message], functions=None, **opts: Any
    ) -> LLMResponse:
        llm_ret = await self._llm.async_chat(messages, functions=functions, stream=False, **opts)
        return LLMResponse(message=llm_ret)

    def _parse_tool_args(self, tool_args: str) -> Dict[str, Any]:
        try:
            args_dict = json.loads(tool_args)
        except json.JSONDecodeError:
            raise ValueError(f"`tool_args` cannot be parsed as JSON. `tool_args`: {tool_args}")

        if not isinstance(args_dict, dict):
            raise ValueError(f"`tool_args` cannot be interpreted as a dict. `tool_args`: {tool_args}")
        return args_dict

    def _init_file_needs_url(self):
        self.file_needs_url = False

        if self._plugins:
            for plugin in self._plugins:
                if plugin not in _PLUGINS_WO_FILE_IO:
                    self.file_needs_url = True

    async def _get_file_manager(self) -> FileManager:
        if self._file_manager is None:
            file_manager = await GlobalFileManagerHandler().get()
        else:
            file_manager = self._file_manager
        return file_manager

    async def _sniff_and_extract_files_from_text( # TODO(shiyutang): to be tested
        self, text: str, plugin_name, file_type: Literal["input", "output"]
    ) -> List[File]:
        files: List[File] = []
        file_ids = extract_file_ids(text)
        for file_id in file_ids:
            file = await self._get_file_from_file_id(file_id, plugin_name, file_type)
            if file is None:
                continue
            else:
                files.append(file)
        return files

    async def _get_file_from_file_id(self, file_id: str, tool: BaseTool, file_type: Literal["input", "output"]) -> Optional[File]:
        if protocol.is_file_id(file_id):
            file_manager = await self._get_file_manager()
            try:
                file = file_manager.look_up_file_by_id(file_id)
            except FileError as e:
                raise FileError(
                    f"Unregistered file with ID {repr(file_id)} is used by {repr(tool)}."
                    f" File type: {file_type}"
                ) from e
            
            return file
