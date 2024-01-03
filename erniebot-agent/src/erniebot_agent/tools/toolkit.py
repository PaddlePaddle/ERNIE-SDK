from typing import List
from erniebot_agent.tools import Tool
from dataclasses import asdict
import uvicorn
import types
import functools
from fastapi import FastAPI

class Toolkit:

    def __init__(self, tools: List[Tool] = None) -> None:
        self.tools: List[Tool] = tools or []
    

    def serve(self, port: int = 5000):
        app = FastAPI(title="erniebot-agent-tools", version="0.0")


        def copy_func(f, func_types, tool):
            # add your code to first parameter
            new_func = types.FunctionType(f.__code__, f.__globals__, f.__name__,
                f.__defaults__, f.__closure__)
            new_func.__annotations__ = func_types
            return functools.partial(new_func, __tool__=tool)

        for tool in self.tools:
            async def create_tool_endpoint_without_inputs(__tool__):
                return await __tool__()

            async def create_tool_endpoint(__tool__, inputs):
                data = asdict(inputs)
                return await __tool__(**data)

            if tool.input_type is not None:
                type_annotation = {"inputs": tool.input_type}
                func = copy_func(create_tool_endpoint, type_annotation, tool)
            else:
                func = copy_func(create_tool_endpoint_without_inputs, {}, tool)
            
            tool_name = tool.tool_name.split("/")[-1]
            app.add_api_route(
                f"/erniebot-agent-tools/0.0/{tool_name}",
                endpoint=func,
                response_model=tool.ouptut_type,
                description=tool.description,
                operation_id=tool.tool_name
            )
        
        @app.get("/.well-known/openapi.yaml")
        def get_openapi_yaml():
            return app.openapi()

        uvicorn.run(app, host="0.0.0.0", port=port)
                