from erniebot_agent.agents.callback.handlers.base import CallbackHandler


class MockCallbackHandler(CallbackHandler):
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
