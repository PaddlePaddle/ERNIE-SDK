from typing import Dict

from erniebot_agent.prompt import PromptTemplate


def generate_agent_role_prompt(agent):
    """Generates the agent role prompt.
    Args: agent (str): The type of the agent.
    Returns: str: The agent role prompt.
    """
    prompts = {
        "Finance Agent": "你是一位经验丰富的金融分析AI助手。你的主要目标是基于提供的数据和趋势，撰写全面、睿智、客观和系统安排的财务报告。",
        "Travel Agent": "你是一位世界旅行的AI导游助手。你的主要任务是撰写有趣、见地深刻、客观和结构良好的旅行报告，包括有关地点的历史、景点和文化见解。",
        "Academic Research Agent": "你是一位AI学术研究助手。你的主要职责是按照学术标准，创建关于特定研究主题的全面、学术严谨、客观和系统化的报告。",
        "Business Analyst": "你是一位经验丰富的AI业务分析助手。你的主要目标是基于提供的业务数据、市场趋势和战略分析，撰写全面、见地深刻、客观和系统化结构的业务报告。",
        "Computer Security Analyst Agent": "你是一位专业从事计算机安全分析的AI。你的主要职责是生成关于计算机安全主题的全面、详细、"
        + "客观和系统化结构的报告，包括漏洞、技术、威胁行为者和高级持久性威胁（APT）组。所有生成的报告应符合学术工作的最高标"
        + "准，并深入剖析计算机安全的复杂性。",
        "Default Agent": "你是一位AI批判性思考研究助手。你的唯一目的是撰写精心写作、广受赞誉、客观和结构化的报告，基于提供的文本。",
    }

    return prompts.get(agent, "No such agent")


def create_message(chunk: str, question: str) -> Dict[str, str]:
    """Create a message for the chat completion

    Args:
        chunk (str): The chunk of text to summarize
        question (str): The question to answer

    Returns:
        Dict[str, str]: The message to send to the chat completion
    """
    return {
        "role": "user",
        "content": f'"""{chunk}""" 使用上述文本，简要回答以下问题："{question}" —— 如果无法使用文本回答问题，'
        "请简要总结文本。"
        "包括所有的事实信息、数字、统计数据等（如果有的话）。字数控制在350字以内。",
    }


def create_abstract(chunk: str) -> Dict[str, str]:
    """Create a message for the chat completion

    Args:
        chunk (str): The chunk of text to summarize
        question (str): The question to answer

    Returns:
        Dict[str, str]: The message to send to the chat completion
    """
    return {
        "role": "user",
        "content": f"""{chunk}，请用中文对上述文章进行总结，总结需要有概括性，不允许输出与文章内容无关的信息，字数控制在500字以内。""",
    }


def create_outline(queries, question):
    ques = ""
    for i, query in enumerate(queries):
        ques += f"{i+1}. {query}\n"
    return {
        "role": "user",
        "content": f"""{ques}，请根据上面的问题生成一个关于"{question}"的大纲，大纲的最后章节是参考文献章节，字数控制在300字以内,并使用json的形式输出。""",
    }


eb_functions = [
    {
        "name": "revise",
        "description": "发送草稿以进行修订",
        "parameters": {
            "type": "object",
            "properties": {
                "notes": {
                    "type": "string",
                    "description": "编辑的中文备注，用于指导修订。",
                },
            },
        },
    },
    {
        "name": "accept",
        "description": "接受草稿",
        "parameters": {
            "type": "object",
            "properties": {"ready": {"const": True}},
        },
    },
]
prompt_markdow_str = """
现在给你1篇报告，你需要判断报告是不是markdown格式，并给出理由。你需要输出判断理由以及判断结果，判断结果是报告是markdown形式或者报告不是markdown格式
你的输出结果应该是个json形式，包括两个键值，一个是"判断理由"，一个是"accept"，如果你认为报告是markdown形式，则"accept"取值为True,如果你认为报告不是markdown形式，则"accept"取值为False，
你需要判断报告是不是markdown格式，并给出理由
{'判断理由':...,'accept':...}
报告：{{report}}
"""
prompt_markdow = PromptTemplate(prompt_markdow_str, input_variables=["report"])
