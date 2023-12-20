from typing import Dict, Generator

import jsonlines

import erniebot


def read_data(json_path):
    list_data = []
    with jsonlines.open(json_path, "r") as f:
        for item in f:
            list_data.append(item)
    return list_data


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


def create_questions(chunk: str, num_questions: int = 5) -> Dict[str, str]:
    """Create a message for the chat completion

    Args:
        chunk (str): The chunk of text to summarize
        question (str): The question to answer

    Returns:
        Dict[str, str]: The message to send to the chat completion
    """
    return {
        "role": "user",
        "content": f"""{chunk}，请根据上面的摘要，生成{num_questions}个问题，问题内容和形式要多样化，分条列举出来.""",
    }


def create_description(chunk: str) -> Dict[str, str]:
    """Create a message for the chat completion

    Args:
        chunk (str): The chunk of text to summarize
        question (str): The question to answer

    Returns:
        Dict[str, str]: The message to send to the chat completion
    """
    return {
        "role": "user",
        "content": f"""{chunk}，请根据上面的摘要，生成一个简短的描述，不超过30字.""",
    }


def split_text(text: str, max_length: int = 8192) -> Generator[str, None, None]:
    """Split text into chunks of a maximum length

    Args:
        text (str): The text to split
        max_length (int, optional): The maximum length of each chunk. Defaults to 8192.

    Yields:
        str: The next chunk of text

    Raises:
        ValueError: If the text is longer than the maximum length
    """
    paragraphs = text.split("\n")
    current_length = 0
    current_chunk = []

    for paragraph in paragraphs:
        if current_length + len(paragraph) + 1 <= max_length:
            current_chunk.append(paragraph)
            current_length += len(paragraph) + 1
        else:
            yield "\n".join(current_chunk)
            current_chunk = [paragraph]
            current_length = len(paragraph) + 1

    if current_chunk:
        yield "\n".join(current_chunk)


def erniebot_chat(
    messages, model="ernie-bot", api_type="aistudio", access_token=None, functions=None, **kwargs
):
    """
    Args:
        messages: dict or list, 输入的消息(message)
        model: str, 模型名称
        api_type: str, 接口类型，可选值包括 'aistudio' 和 'qianfan'
        access_token: str, 访问令牌(access token)
        functions: list, 函数列表
        kwargs: 其他参数

    Returns:
        dict or list, 返回聊天结果
    """
    _config = dict(
        api_type=api_type,
        access_token=access_token,
    )
    if functions is None:
        resp_stream = erniebot.ChatCompletion.create(
            _config_=_config, model=model, messages=messages, **kwargs
        )
    else:
        resp_stream = erniebot.ChatCompletion.create(
            _config_=_config, model=model, messages=messages, **kwargs, functions=functions
        )
    return resp_stream
