import unittest

class TestCalculator(unittest.TestCase):
    def run_query(self, query):
        response = erniebot.ChatCompletion.create(
            model='ernie-bot',
            messages=[{
                'role': 'user',
                'content': few_shot_prompt + query,
            }, ],
            functions=[

            ],
            stream=False)

        result = response.get_result()
        print(colored("get result from ebbot: " + result, "light_green"))
        json_result = result.strip("输出：").strip().strip("```json").strip("```").strip()
        a = json.loads(json_result)
        command = a["arguments"]["command"]
        result = calculator_compute(command)
        print(result)
        return result
    
    def test_add(self):
        result = self.run_query("3 加四等于多少")
        self.assertEqual(result, 7)
    
    def test_multiply(self):
        result = self.run_query("3乘以五等于多少")
        self.assertEqual(result, 15)

    def test_complex_formula(self):
        result = self.run_query("3乘以五 再加10等于多少")
        self.assertEqual(result, 25)
