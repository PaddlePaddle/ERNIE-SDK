import unittest

from erniebot_agent.prompts.prompt_template import PromptTemplate


class TestPrompt(unittest.TestCase):
    def setUp(self) -> None:
        self.template = "请回答下列问题，如果不知道就回答不知道：{{query}}"
        self.prompt_template = PromptTemplate(
                        template=self.template,
                        input_variables=['query'])
    
    def test_prompt(self) -> None:
        """test contruct message from prompt"""
        prompt = self.prompt_template.format(query="天上的星星有多少颗?")

        self.assertEqual(prompt, '请回答下列问题，如果不知道就回答不知道：天上的星星有多少颗?')  

    def test_prompt_with_agent(self) -> None:
        """test contruct message from prompt with agent"""
        pass 
        # Todo when agent is implemented
        # prompt = self.prompt_template.format(query="地球上的人口有多少？请精确到个位数。")
        # message = UserMessage(prompt)

if __name__ == '__main__':
    unittest.main()