import logging
import re
from typing import Any, List, Optional

from tools.utils import JsonUtil, ReportCallbackHandler

from erniebot_agent.chat_models.erniebot import BaseERNIEBot
from erniebot_agent.memory import HumanMessage, Message, SystemMessage
from erniebot_agent.prompt import PromptTemplate

logger = logging.getLogger(__name__)
PLAN_VERIFICATIONS_PROMPT = """
为了验证给出的内容中数字性表述的准确性，您需要创建一系列验证问题，
用于测试原始基线响应中的事实主张。例如，如果长格式响应的一部分包含
“墨西哥-美国战争是 1846 年至 1848 年美国和墨西哥之间的武装冲突”，
那么一种可能的验证问题可以是：“墨西哥-美国战争何时开始以及结束？”
给定内容：{{base_context}}
您需要按照列表输出，输出段落中的事实和相应的验证问题。
[{"fact": <段落中的事实>, "question": <验证问题，通过结合查询和事实生成>},
{"fact": <段落中的事实>, "question": <验证问题，通过结合查询和事实生成>}, ...]
"""
ANWSER_PROMPT = """
根据外部知识回答问题。如果给出的外部知识不能解答提出的问题，
请直接输出"无法回答"，无需提供答案。给定问题:\n{{question}}\n
外部知识:{{content}}\n回答:
"""
CHECK_CLAIM_PROMPT = """
根据给定的问题和回答，判断事实中数字描述是否正确。如果认为事实中的数字描述不正确，
请根据问题和回答提供修正。您的输出应为 JSON 格式：
{"is_correct": <事实是否正确>, "modify": <对不正确的事实进行修正>}
给定问题: {{question}}\n回答: {{answer}}\n事实: {{claim}}
"""

FINAL_RESPONSE_PROMPT = """
根据提供的背景知识，对原始内容进行改写。确保改写后的内容中的数字与背景知识中的数字一致。
您必须修正原始内容中的数字。原始内容：{{origin_content}}\n背景知识：{{context}}
改写内容：
"""


class FactCheckerAgent(JsonUtil):
    DEFAULT_SYSTEM_MESSAGE = "你是一个事实检查助手，你的任务就是检查文本中的事实描述是否正确"

    def __init__(
        self,
        name: str,
        llm: BaseERNIEBot,
        retriever_db: Any,
        system_message: Optional[SystemMessage] = None,
        callbacks=None,
    ):
        """
        Initialize a fact_checker agent.

        args:
            name: The name of the agent.
            llm: An LLM for the agent to use.
            retriever_db: A database for the agent to use.
            system_message: A message to be displayed when the agent starts.
            callbacks: A callback handler for the agent.
        """
        self.name = name
        self.llm = llm
        self.retriever_db = retriever_db
        self.prompt_plan_verifications = PromptTemplate(
            PLAN_VERIFICATIONS_PROMPT, input_variables=["base_context"]
        )
        self.prompt_anwser = PromptTemplate(ANWSER_PROMPT, input_variables=["question", "content"])
        self.prompt_check_claim = PromptTemplate(
            CHECK_CLAIM_PROMPT, input_variables=["question", "answer", "claim"]
        )
        self.prompt_final_response = PromptTemplate(
            FINAL_RESPONSE_PROMPT, input_variables=["origin_content", "context"]
        )
        self.system_message = (
            system_message.content if system_message is not None else self.DEFAULT_SYSTEM_MESSAGE
        )
        if callbacks is None:
            self._callback_manager = ReportCallbackHandler()
        else:
            self._callback_manager = callbacks

    async def run(self, report: str):
        """
        The main logic of running the agent.

        Args:
            report: Entered report text.
        Returns:
            The results of the agent's operation.
        """
        await self._callback_manager.on_run_start(
            agent=self, agent_name=self.name, prompt=self.system_message
        )
        agent_resp = await self._run(report=report)
        await self._callback_manager.on_run_end(agent=self, response=agent_resp)
        return agent_resp

    async def generate_anwser(self, question: str, context: str):
        """
        Generate answers to questions based on background knowledge

        Args:
            question: Indicates a question to be answered.
            context: Represents background knowledge relevant to the problem.

        Returns:
            Generated answers to questions.
        """
        messages: List[Message] = [
            HumanMessage(content=self.prompt_anwser.format(question=question, content=context))
        ]
        responese = await self.llm.chat(messages)
        result = responese.content
        return result

    async def check_claim(self, question: str, answer: str, claim: str):
        """
        Use a large language model to conduct a conversation, verify a fact,
        and suggest corrections if the fact is incorrect

        Args:
            question: represents a fact-checking question
            answer: represents a fact-checking answer
            claim: indicates a fact that need to be verified

        Returns:
            A dictionary containing verification results,
        including whether the facts are correct and suggestions for correction.
        """
        messages: List[Message] = [
            HumanMessage(
                content=self.prompt_check_claim.format(question=question, answer=answer, claim=claim)
            )
        ]
        responese = await self.llm.chat(messages)
        result = responese.content
        result = self.parse_json(result)
        return result

    async def verifications(self, facts_problems: List[dict]):
        """
        Answer questions using external knowledge and then use the answers to the questions to verify
        relevant facts. As it processes each question and fact pair, it obtains the context
        relevant to the question, generates an answer to the question, checks whether the fact
        is correct, and records the verification results.

        Args:
            facts_problems: A list of dictionaries containing questions and related facts.

        Returns:
            Updated dictionary list of verified questions and facts.
        """
        for item in facts_problems:
            question = item["question"]
            claim = item["fact"]
            context = self.retriever_db.search(question)
            context = [i["content"] for i in context]
            item["evidence"] = context
            anwser = await self.generate_anwser(question, context)
            item["anwser"] = anwser
            result = await self.check_claim(question, anwser, claim)
            item["is_correct"] = result["is_correct"]
            if result["is_correct"] is False:
                item["modify"] = result["modify"]
            else:
                item["modify"] = claim
        self._callback_manager._agent_info(msg=item["modify"], subject="事实验证的结果", state="End")
        return facts_problems

    async def generate_final_response(self, content: str, verifications: List[dict]):
        """
        If the original factual questions pass fact verification,
        the original content will be returned directly.
        Otherwise, the original content will be corrected based
        on the results of factual verification.

        Args:
            content: Original text content.
            verifications: List of dictionaries containing fact verification results.

        Returns:
            The final generated reply content.
        """
        if all([item["is_correct"] for item in verifications]):
            return content
        else:
            context = "".join([item["modify"] for item in verifications])
            messages: List[Message] = [
                HumanMessage(
                    content=self.prompt_final_response.format(origin_content=content, context=context)
                )
            ]
            resulte = await self.llm.chat(messages)
            result = resulte.content
            return result

    async def report_fact(self, report: str):
        """
        Filter out sentences containing numbers in text.
        Extract facts from the filtered sentences.
        Extract validation questions and verify each extracted fact.
        Example:
        "Mexican-American War
        was the armed conflict between the United States and Mexico from 1846 to 1848," then one possibility
        A validation question to check these dates could be When did the Mexican-American War begin and end?

        Args:
            report: The original text content.

        Returns:
            The final generated reply content.
        """
        report_list = report.split("\n\n")
        text = []
        for item in report_list:
            if item.strip()[0] == "#":
                text.append(item)
            else:
                contains_numbers = re.findall(r"\b\d+\b", item)
                if contains_numbers:
                    messages: List[Message] = [
                        HumanMessage(content=self.prompt_plan_verifications.format(base_context=item))
                    ]
                    responese = await self.llm.chat(messages)
                    result: List[dict] = self.parse_json(responese.content)
                    fact_check_result: List[dict] = await self.verifications(result)
                    new_item: str = await self.generate_final_response(item, fact_check_result)
                    text.append(new_item)
                else:
                    text.append(item)
        return "\n\n".join(text)

    async def _run(self, report: str):
        """
        The main logic of running the agent.

        Args:
            report: Entered report text.
        Returns:
            Processed report text.
        """
        report = await self.report_fact(report)
        return report
