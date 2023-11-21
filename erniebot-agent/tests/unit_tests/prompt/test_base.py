import unittest

from erniebot_agent.messages import HumanMessage
from erniebot_agent.prompt.prompt_template import PromptTemplate


class TestPrompt(unittest.TestCase):
    def setUp(self) -> None:
        self.template = "请回答下列问题，如果不知道就回答不知道：{{query}}"
        self.prompt_template = PromptTemplate(template=self.template, input_variables=["query"])

    def test_prompt(self) -> None:
        """test contruct message from prompt"""
        prompt = self.prompt_template.format(query="天上的星星有多少颗?")

        self.assertEqual(prompt, "请回答下列问题，如果不知道就回答不知道：天上的星星有多少颗?")

    def test_prompt_with_except_keys(self) -> None:
        """test contruct message from prompt with except keys"""
        self.prompt_template = PromptTemplate(template=self.template, input_variables=["querys"])

        with self.assertRaises(KeyError):
            self.prompt_template.format(query="天上的星星有多少颗?")

    def test_prompt_with_message_output(self) -> None:
        prompt = self.prompt_template.format_as_message(message_class=HumanMessage, query="天上的星星有多少颗?")
        self.assertTrue(isinstance(prompt, HumanMessage))


if __name__ == "__main__":
    unittest.main()
