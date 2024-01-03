from typing import Dict
from erniebot_agent.prompt import PromptTemplate


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


def generate_search_queries_prompt(question):
    """Generates the search queries prompt for the given question.
    Args: question (str): The question to generate the search queries prompt for
    Returns: str: The search queries prompt for the given question
    """
    queries_prompt = """
    写出 4 个谷歌搜索查询，以从以下内容形成客观意见： "{{question}}"
    您必须以以下格式回复一个中文字符串列表：["query 1", "query 2", "query 3", "query 4"].
    """
    Queries_prompt = PromptTemplate(queries_prompt, input_variables=["question"])
    return Queries_prompt.format(question=question)


def generate_search_queries_with_context(context, question):
    """Generates the search queries prompt for the given question.
    Args: question (str): The question to generate the search queries prompt for
    Returns: str: The search queries prompt for the given question
    """
    queries_prompt = """
    {{context}} 根据上述信息，写出 4 个搜索查询，以从以下内容形成客观意见： "{{question}}"
    您必须以以下格式回复一个中文字符串列表：["query 1", "query 2", "query 3", "query 4"].
    """
    Queries_prompt = PromptTemplate(queries_prompt, input_variables=["context", "question"])
    return Queries_prompt.format(context=context, question=question)


def generate_search_queries_with_context_comprehensive(context, question):
    """Generates the search queries prompt for the given question.
    Args: question (str): The question to generate the search queries prompt for
    Returns: str: The search queries prompt for the given question
    """
    context_comprehensive = """
    你的任务是根据给出的多篇context内容，综合考虑这些context的内容，写出4个综合性搜索查询。现在多篇context为{{context}}
    你需要综合考虑上述信息，写出 4 个综合性搜索查询，以从以下内容形成客观意见： "{{question}}"
    您必须以以下格式回复一个中文字符串列表：["query 1", "query 2", "query 3", "query 4"]。
    """
    prompt = PromptTemplate(context_comprehensive, input_variables=["context", "question"])
    return prompt.format(context=str(context), question=question)


def generate_report_prompt(question, research_summary, outline=None):
    """Generates the report prompt for the given question and research summary.
    Args: question (str): The question to generate the report prompt for
            research_summary (str): The research summary to generate the report prompt for
    Returns: str: The report prompt for the given question and research summary
    """
    if outline is None:
        report_prompt = """你是任务是生成一份满足要求的报告，
        [请你注意生成的报告第一行必须是报告的题目，报告题目是一级标题，前面要有'#'标志符。]
        报告的格式必须是markdown格式，注意报告标题前面必须有'#'
        现在给你一些信息，帮助你进行报告生成任务。
        信息：{{information}}
        使用上述信息，详细报告回答以下问题或主题{{question}}
        -----
        报告应专注于回答问题，结构良好，内容丰富，包括事实和数字（如果有的话），字数控制在3000字，并采用Markdown语法和APA格式。
        注意报告标题前面必须有'#'。
        请你注意生成的报告第一行必须是报告的题目，并且报告题目是一级标题，前面要有'#'标志符。
        您必须基于给定信息确定自己的明确和有效观点。不要得出一般和无意义的结论。
        在报告末尾以APA格式列出所有使用的来源URL。

        """
        Report_prompt = PromptTemplate(report_prompt, input_variables=["information", "question"])
        strs = Report_prompt.format(information=research_summary, question=question)
    else:
        outline = outline.replace('"', "'")
        report_prompt = """你是任务是生成一份满足要求的报告，报告的格式必须是markdown格式，注意报告标题前面必须有'#'
        现在给你一些信息，帮助你进行报告生成任务
        信息：{{information}}
        使用上述信息，根据设定好的大纲{{outline}}
        详细报告回答以下问题或主题{{question}}
        -----
        报告应专注于回答问题，结构良好，内容丰富，包括事实和数字（如果有的话），字数控制在3000字，并采用Markdown语法和APA格式。
        注意报告标题前面必须有'#'
        您必须基于给定信息确定自己的明确和有效观点。不要得出一般和无意义的结论。
        在报告末尾以APA格式列出所有使用的来源URL。
        """
        Report_prompt = PromptTemplate(report_prompt, input_variables=["information", "outline", "question"])
        strs = Report_prompt.format(information=research_summary, outline=outline, question=question)
    return strs.replace(". ", ".")


def generate_resource_report_prompt(question, research_summary, **kwargs):
    """Generates the resource report prompt for the given question and research summary.

    Args:
        question (str): The question to generate the resource report prompt for.
        research_summary (str): The research summary to generate the resource report prompt for.

    Returns:
        str: The resource report prompt for the given question and research summary.
    """
    report_prompt = """
    {{information}}根据上述信息，为以下问题或主题生成一份参考文献推荐报告"{{question}}"。
    该报告应详细分析每个推荐的资源，解释每个来源如何有助于找到研究问题的答案。
    着重考虑每个来源的相关性、可靠性和重要性。确保报告结构良好，信息丰富，深入，并遵循Markdown语法。
    在可用时包括相关的事实、数字和数据。报告的最低长度应为1,200字。
    """
    Report_prompt = PromptTemplate(report_prompt, input_variables=["information", "question"])
    strs = Report_prompt.format(information=research_summary, question=question)
    return strs.replace(". ", ".")


def generate_outline_report_prompt(question, research_summary, **kwargs):
    """Generates the outline report prompt for the given question and research summary.
    Args: question (str): The question to generate the outline report prompt for
            research_summary (str): The research summary to generate the outline report prompt for
    Returns: str: The outline report prompt for the given question and research summary
    """
    report_prompt = """{{information}}使用上述信息，为以下问题或主题：
    "{{question}}". 生成一个Markdown语法的研究报告大纲。
    大纲应为研究报告提供一个良好的结构框架，包括主要部分、子部分和要涵盖的关键要点。
    研究报告应详细、信息丰富、深入，至少1,200字。使用适当的Markdown语法来格式化大纲，确保可读性。
    """
    Report_prompt = PromptTemplate(report_prompt, input_variables=["information", "question"])
    strs = Report_prompt.format(information=research_summary, question=question)
    return strs.replace(". ", ".")


def get_report_by_type(report_type):
    report_type_mapping = {
        "research_report": generate_report_prompt,
        "resource_report": generate_resource_report_prompt,
        "outline_report": generate_outline_report_prompt,
    }
    return report_type_mapping[report_type]


EDIT_TEMPLATE = """你是一名编辑。
你被指派任务编辑以下草稿，该草稿由一名非专家撰写。
如果这份草稿足够好以供发布，请接受它，或者将它发送进行修订，同时附上指导修订的笔记。
你应该检查以下事项：
- 这份草稿必须充分回答原始问题。
- 这份草稿必须按照APA格式编写。
- 这份草稿必须不包含低级的句法错误。
如果不符合以上所有标准，你应该发送适当的修订笔记。
"""

EB_EDIT_TEMPLATE = """你是一名编辑。
你被指派任务编辑以下草稿，该草稿由一名非专家撰写。
如果这份草稿足够好以供发布，请接受它，或者将它发送进行修订，同时附上指导修订的笔记。
你应该检查以下事项：
- 这份草稿必须充分回答原始问题。
- 这份草稿必须按照APA格式编写。
- 这份草稿必须不包含低级的句法错误。
- 这份草稿的标题不能包含任何引用
如果不符合以上所有标准，你应该发送适当的修订笔记，请以json的格式输出：
如果需要进行修订，则按照下面的格式输出：{"accept":"false","notes": "分条列举出来所给的修订建议。"} 否则输出： {"accept": "true","notes":""}
"""


def generate_revisor_prompt(draft, notes):
    return f"""你是一名专业作家。你已经受到编辑的指派，需要修订以下草稿，该草稿由一名非专家撰写。你可以选择是否遵循编辑的备注，视情况而定。
            使用中文输出，只允许对草稿进行局部修改，不允许对草稿进行胡编乱造。
            草稿:\n\n{draft}" + "编辑的备注:\n\n{notes}
            """


def rank_report_prompt(report, query):
    prompt_socre = """现在给你1篇报告，现在你需要严格按照以下的标准，对这个报告进行打分，越符合标准得分越高，打分区间在0-10之间，
    你输出的应该是一个json格式，json中的键值为"打分理由"和"报告总得分"，{'打分理由':...,'报告总得分':...}
    对报告进行打分,打分标准如下：
    1.仔细检查报告格式，报告必须是完整的，包括标题、摘要、正文、参考文献等，完整性越高，得分越高，这一点最高给4分。
    3.仔细检查报告内容，报告内容与{{query}}问题相关性越高得分越高，这一点最高给4分。
    4.仔细检查报告格式，标题是否有"#"符号标注，这一点最高给2分，没有"#"给0分，有"#"给1分。
    5.仔细检查报告格式，报告的标题句结尾不能有任何中文符号，标题结尾有中文符号给0分，标题结尾没有中文符号给1分。
    以下是这篇报告的内容：{{content}}
    请你记住，你需要根据打分标准给出每篇报告的打分理由，打分理由报告
    最后给出打分结果和最终的打分列表。
    你的输出需要按照以下格式进行输出：
    为了对这报告进行打分，我将根据给定的标准进行评估。报告的打分理由将基于以下五个标准：
    1) 是否包含标题、摘要、正文、参考文献等，3) 内容与问题的相关性，4) 标题是否有"#"标注，5) 标题是否有中文符号。
    """
    Prompt_socre = PromptTemplate(prompt_socre, input_variables=["query", "content"])
    strs = Prompt_socre.format(content=report, query=query)
    return strs


def filter_report(report):
    return f"""{report},对上述报告进行评估，你应该检查以下的各项：
           1.报告必须是完整的，包括标题、摘要、正文、参考文献等。否则返回False。
           2.报告必须是Markdown语法，并且标题有"#"符号标注。否则返回False。
           如果都符合要求，则返回True，只回复评估结果，不用言语或解释。
           """


def evaluate_report(report, query):
    return f"""{report}, 查询={query}，根据上述信息，对报告和给定查询的相关性以及报告的格式等进行评估。
            你应该检查以下事项：
                - 这份草稿必须充分回答原始问题。
                - 这份草稿必须不包含低级的句法错误。
            输出格式为 [True, False, True]，其中True表示相关，False表示不相关，只回复结果，不用言语或解释。
            """


def generate_reference(meta_dict):
    json_format = """{
            "参考文献": [
                {
                "标题": "文章标题",
                "链接": "文章链接",
                }]
            }"""
    return (
        f""""{meta_dict},根据上面的数据，生成报告的参考文献，按照如下json的形式输出:
            """
        + json_format
    )


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
