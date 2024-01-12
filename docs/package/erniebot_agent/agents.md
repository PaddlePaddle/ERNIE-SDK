
# Agent Module


::: erniebot_agent.agents
    options:
        summary: true
        separate_signature: true
        show_signature_annotations: true
        line_length: 60
        members:
        - Agent
        - FunctionAgent

::: erniebot_agent.agents.callback
    options:
        summary: true
        separate_signature: true
        show_signature_annotations: true
        line_length: 60
        members:
        - CallbackManager
        - CallbackHandler
        - LoggingHandler

::: erniebot_agent.agents.schema
    options:
        summary: true
        separate_signature: true
        show_signature_annotations: true
        line_length: 60
        members:
        - LLMResponse
        - ToolResponse
        - AgentResponse
