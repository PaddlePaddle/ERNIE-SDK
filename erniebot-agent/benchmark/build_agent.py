import os
from typing import Any, Dict

os.environ["EB_AGENT_ACCESS_TOKEN"] = "4ce50e3378f418d271c480c8ddfa818537071dbe"
os.environ["EB_AGENT_LOGGING_LEVEL"] = "info"

from benchmark.schema import AgentArgs
from benchmark.utils import relative_import


class AgentBuilder(object):
    """The builder of the agent."""

    def __init__(self, args: AgentArgs):
        self.args = args
        self.config_check()

        self.model = self._build_model(args["model"], args["root_path"]["model"])
        self.memory = self._build_memory(args["memory"], args["root_path"]["memory"])
        self.tools = self._build_toolset(args["tools"], args["root_path"]["tools"]) if args["tools"] else []
        self.agent = self._build_agent(args["agent"], args["root_path"]["agent"])

    def config_check(self):
        root_path_include = ["model", "memory", "tools", "agent"]
        for module_name in root_path_include:
            if module_name not in self.args["root_path"]:
                raise ValueError(f"{module_name} module path is not specified in the root path.")

            if module_name not in self.args:
                raise ValueError(f"{module_name} config is not specified in the config file.")

    def _build_model(self, model_args: Dict[str, Any], root_path):
        """Build the model."""
        model = relative_import(root_path, model_args["name"])
        return model(**model_args["kwargs"])

    def _build_memory(self, memory_args: Dict[str, Any], root_path):
        """Build the memory."""
        memory = relative_import(root_path, memory_args["name"])
        return memory(**memory_args["kwargs"])

    def _build_toolset(self, tools: Dict[str, Any], root_path):
        """Build the toolset."""
        # NOTE: Tool initialization does not need params.
        toolset = []
        for tool_arg in tools:
            tool = relative_import(root_path, tool_arg["name"])
            toolset.append(tool(**tool_arg["kwargs"]))
        return toolset

    def _build_agent(self, agent_args, root_path):
        """Build the agent."""
        # NOTE: Other way of passing in params is not included.
        agent = relative_import(root_path, agent_args["name"])
        return agent(llm=self.model, memory=self.memory, tools=self.tools, **agent_args["kwargs"])
