from erniebot_agent.agents.callback.handlers.base import CallbackHandler
from erniebot_agent.tools.base import Tool, ToolParameterView
from pydantic import Field


class CountingCallbackHandler(CallbackHandler):
    def __init__(self):
        super().__init__()
        self.run_starts = 0
        self.llm_starts = 0
        self.llm_ends = 0
        self.llm_errors = 0
        self.tool_starts = 0
        self.tool_ends = 0
        self.tool_errors = 0
        self.run_ends = 0

    async def on_run_start(self, agent, prompt):
        self.run_starts += 1

    async def on_llm_start(self, agent, llm, messages):
        self.llm_starts += 1

    async def on_llm_end(self, agent, llm, response):
        self.llm_ends += 1

    async def on_llm_error(self, agent, llm, error):
        self.llm_errors += 1

    async def on_tool_start(self, agent, tool, input_args):
        self.tool_starts += 1

    async def on_tool_end(self, agent, tool, response):
        self.tool_ends += 1

    async def on_tool_error(self, agent, tool, error):
        self.tool_errors += 1

    async def on_run_end(self, agent, response):
        self.run_ends += 1


class IdentityTool(Tool):
    class _InputView(ToolParameterView):
        input: str = Field(description="输入字符串")

    class _OutputView(ToolParameterView):
        identity: float = Field(description="输出字符串，与输入字符串相同")

    description = "该工具原样返回输入字符串"
    input_type = _InputView
    ouptut_type = _OutputView

    async def __call__(self, input):
        return {"identity": input}
