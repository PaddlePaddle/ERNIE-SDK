import abc
import json
from typing import (
    Any,
    Dict,
    Final,
    Iterable,
    List,
    Mapping,
    NoReturn,
    Optional,
    Sequence,
    Tuple,
    Union,
    final,
)

from erniebot_agent.agents.base import BaseAgent
from erniebot_agent.agents.callback.callback_manager import CallbackManager
from erniebot_agent.agents.callback.default import get_default_callbacks
from erniebot_agent.agents.callback.handlers.base import CallbackHandler
from erniebot_agent.agents.mixins import GradioMixin
from erniebot_agent.agents.schema import AgentResponse, LLMResponse, ToolResponse
from erniebot_agent.chat_models.erniebot import BaseERNIEBot
from erniebot_agent.file import (
    File,
    FileManager,
    GlobalFileManagerHandler,
    get_default_file_manager,
)
from erniebot_agent.memory import Memory, WholeMemory
from erniebot_agent.memory.messages import Message, SystemMessage
from erniebot_agent.tools.base import BaseTool
from erniebot_agent.tools.tool_manager import ToolManager
from erniebot_agent.utils.exceptions import FileError

_PLUGINS_WO_FILE_IO: Final[Tuple[str]] = ("eChart",)


class Agent(GradioMixin, BaseAgent[BaseERNIEBot]):
    """The base class for agents.

    Typically, this class should be the base class for custom agent classes. A
    class derived from this class must implement how the agent orchestates the
    components to complete tasks.

    Attributes:
        llm: The LLM that the agent uses.
        memory: The message storage that keeps the chat history.
    """

    llm: BaseERNIEBot
    memory: Memory

    def __init__(
        self,
        llm: BaseERNIEBot,
        tools: Union[ToolManager, Iterable[BaseTool]],
        *,
        memory: Optional[Memory] = None,
        system_message: Optional[SystemMessage] = None,
        callbacks: Optional[Union[CallbackManager, Iterable[CallbackHandler]]] = None,
        file_manager: Optional[FileManager] = None,
        plugins: Optional[List[str]] = None,
    ) -> None:
        """Initialize an agent.

        Args:
            llm: An LLM for the agent to use.
            tools: Tools for the agent to use.
            memory: A memory object that equips the agent to remember chat
                history. If `None`, a `WholeMemory` object will be used.
            system_message: A message that tells the LLM how to interpret the
                conversations. If `None`, the system message contained in
                `memory` will be used.
            callbacks: Callback handlers for the agent to use. If `None`, a
                default list of callbacks will be used.
            file_manager: A file manager for the agent to interact with files.
                If `None`, a global file manager that can be shared among
                different components will be implicitly created and used.
            plugins: A list of names of the plugins for the agent to use. If
                `None`, the agent will use a default list of plugins. Set
                `plugins` to `[]` to disable the use of plugins.
        """
        super().__init__()
        self.llm = llm
        if isinstance(tools, ToolManager):
            self._tool_manager = tools
        else:
            self._tool_manager = ToolManager(tools)
        if memory is None:
            memory = self._create_default_memory()
        self.memory = memory
        if system_message:
            self.system_message = system_message
        else:
            self.system_message = self.memory.get_system_message()
        if callbacks is None:
            callbacks = get_default_callbacks()
        if isinstance(callbacks, CallbackManager):
            self._callback_manager = callbacks
        else:
            self._callback_manager = CallbackManager(callbacks)
        self._file_manager = file_manager or get_default_file_manager()
        if plugins is not None:
            raise ValueError("Currently, plugins are not supported.")
        self._plugins = plugins
        if plugins is not None:
            raise NotImplementedError("The use of plugins is not supported yet.")
        self._init_file_needs_url()

    @final
    async def run(self, prompt: str, files: Optional[Sequence[File]] = None) -> AgentResponse:
        """Run the agent asynchronously.

        Args:
            prompt: A natural language text describing the task that the agent
                should perform.
            files: A list of files that the agent can use to perform the task.

        Returns:
            Response from the agent.
        """
        if files:
            await self._ensure_managed_files(files)
        await self._callback_manager.on_run_start(agent=self, prompt=prompt)
        try:
            agent_resp = await self._run(prompt, files)
        except BaseException as e:
            await self._callback_manager.on_run_error(agent=self, error=e)
            raise e
        else:
            await self._callback_manager.on_run_end(agent=self, response=agent_resp)
        return agent_resp

    @final
    async def run_llm(
        self,
        messages: List[Message],
        *,
        llm_opts: Optional[Mapping[str, Any]] = None,
    ) -> LLMResponse:
        """Run the LLM asynchronously.

        Args:
            messages: The input messages.
            llm_opts: Options to pass to the LLM.

        Returns:
            Response from the LLM.
        """
        await self._callback_manager.on_llm_start(agent=self, llm=self.llm, messages=messages)
        try:
            llm_resp = await self._run_llm(messages, **(llm_opts or {}))
        except (Exception, KeyboardInterrupt) as e:
            await self._callback_manager.on_llm_error(agent=self, llm=self.llm, error=e)
            raise e
        else:
            await self._callback_manager.on_llm_end(agent=self, llm=self.llm, response=llm_resp)
        return llm_resp

    @final
    async def run_tool(self, tool_name: str, tool_args: str) -> ToolResponse:
        """Run the specified tool asynchronously.

        Args:
            tool_name: The name of the tool to run.
            tool_args: The tool arguments in JSON format.

        Returns:
            Response from the tool.
        """
        tool = self._tool_manager.get_tool(tool_name)
        await self._callback_manager.on_tool_start(agent=self, tool=tool, input_args=tool_args)
        try:
            tool_resp = await self._run_tool(tool, tool_args)
        except (Exception, KeyboardInterrupt) as e:
            await self._callback_manager.on_tool_error(agent=self, tool=tool, error=e)
            raise e
        else:
            await self._callback_manager.on_tool_end(agent=self, tool=tool, response=tool_resp)
        return tool_resp

    def load_tool(self, tool: BaseTool) -> None:
        """Load a tool into the agent.

        Args:
            tool: The tool to load.
        """
        self._tool_manager.add_tool(tool)

    def unload_tool(self, tool: Union[BaseTool, str]) -> None:
        """Unload a tool from the agent.

        Args:
            tool: The tool to unload.
        """
        if isinstance(tool, str):
            tool = self.get_tool(tool)
        self._tool_manager.remove_tool(tool)

    def get_tool(self, tool_name: str) -> BaseTool:
        """Get a tool by name."""
        return self._tool_manager.get_tool(tool_name)

    def get_tools(self) -> List[BaseTool]:
        """Get the tools that the agent can choose from."""
        return self._tool_manager.get_tools()

    def reset_memory(self) -> None:
        """Clear the chat history."""
        self.memory.clear_chat_history()

    def get_file_manager(self) -> FileManager:
        # Can we create a lazy proxy for the global file manager and simply set
        # and use `self._file_manager`?
        if self._file_manager is None:
            file_manager = GlobalFileManagerHandler().get()
        else:
            file_manager = self._file_manager
        return file_manager

    @abc.abstractmethod
    async def _run(self, prompt: str, files: Optional[Sequence[File]] = None) -> AgentResponse:
        raise NotImplementedError

    async def _run_llm(self, messages: List[Message], **opts: Any) -> LLMResponse:
        for reserved_opt in ("stream", "functions", "system", "plugins"):
            if reserved_opt in opts:
                raise TypeError(f"`{reserved_opt}` should not be set.")
        functions = self._tool_manager.get_tool_schemas()
        opts["system"] = self.system_message.content if self.system_message is not None else None
        opts["plugins"] = self._plugins
        llm_ret = await self.llm.chat(messages, stream=False, functions=functions, **opts)
        return LLMResponse(message=llm_ret)

    async def _run_tool(self, tool: BaseTool, tool_args: str) -> ToolResponse:
        parsed_tool_args = self._parse_tool_args(tool_args)
        file_manager = self.get_file_manager()
        # XXX: Sniffing is less efficient and probably unnecessary.
        # Can we make a protocol to statically recognize file inputs and outputs
        # or can we have the tools introspect about this?
        input_files = file_manager.sniff_and_extract_files_from_list(list(parsed_tool_args.values()))
        tool_ret = await tool(**parsed_tool_args)
        if isinstance(tool_ret, dict):
            output_files = file_manager.sniff_and_extract_files_from_list(list(tool_ret.values()))
        else:
            output_files = []
        tool_ret_json = json.dumps(tool_ret, ensure_ascii=False)
        return ToolResponse(json=tool_ret_json, input_files=input_files, output_files=output_files)

    def _create_default_memory(self) -> Memory:
        return WholeMemory()

    def _init_file_needs_url(self):
        self.file_needs_url = False
        if self._plugins:
            for plugin in self._plugins:
                if plugin not in _PLUGINS_WO_FILE_IO:
                    self.file_needs_url = True

    def _parse_tool_args(self, tool_args: str) -> Dict[str, Any]:
        try:
            args_dict = json.loads(tool_args)
        except json.JSONDecodeError:
            raise ValueError(f"`tool_args` cannot be parsed as JSON. `tool_args`: {tool_args}")

        if not isinstance(args_dict, dict):
            raise ValueError(f"`tool_args` cannot be interpreted as a dict. `tool_args`: {tool_args}")
        return args_dict

    async def _ensure_managed_files(self, files: Sequence[File]) -> None:
        def _raise_exception(file: File) -> NoReturn:
            raise FileError(f"{repr(file)} is not managed by the file manager of the agent.")

        file_manager = self.get_file_manager()
        for file in files:
            try:
                managed_file = file_manager.look_up_file_by_id(file.id)
            except FileError:
                _raise_exception(file)
            if file is not managed_file:
                _raise_exception(file)
