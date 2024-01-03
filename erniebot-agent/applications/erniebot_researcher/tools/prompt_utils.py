from typing import Dict


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
