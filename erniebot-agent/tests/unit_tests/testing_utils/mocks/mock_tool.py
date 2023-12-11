from erniebot_agent.tools.base import BaseTool


class FakeTool(BaseTool):
    def __init__(self, name, description, parameters, responses, function):
        super().__init__()
        self.name = name
        self.description = description
        self.schema = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "responses": responses,
        }
        self.function = function

    @property
    def tool_name(self):
        return self.name

    @property
    def examples(self):
        return []

    async def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)

    def function_call_schema(self):
        return self.schema
