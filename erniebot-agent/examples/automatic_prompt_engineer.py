import argparse
import asyncio

from erniebot_agent.agents.prompt_agent import PromptAgent
from erniebot_agent.chat_models import ERNIEBot
from erniebot_agent.memory import WholeMemory
from erniebot_agent.tools.openai_search_tool import OpenAISearchTool
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from prettytable import PrettyTable
from utils import (
    create_description,
    create_keywords,
    create_questions,
    erniebot_chat,
    read_data,
)

import erniebot

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--api_type", default=None, type=str, help="The API Key.")
parser.add_argument("--access_token", default=None, type=str, help="The secret key.")
parser.add_argument("--summarization_path", default='data/data.jsonl', type=str, help="The output path.")
parser.add_argument("--number_of_prompts", default=3, type=int, help="The number of tool descriptions.")
parser.add_argument("--num_questions", default=3, type=int, help="The number of few shot questions.")
parser.add_argument("--num_keywords", default=-1, type=int, help="The number of few shot questions.")
args = parser.parse_args()
# yapf: enable


def generate_candidate_prompts(description, number_of_prompts):
    prompts = []
    for i in range(number_of_prompts):
        messages = [create_description(description)]
        results = erniebot_chat(
            messages, model="ernie-bot-4", api_type=args.api_type, access_token=args.access_token
        )["result"]
        prompts.append(results)
    return prompts


def generate_candidate_questions(
    description, num_questions: int = -1, num_keywords: int = -1, temperature=1e-10
):
    if num_questions > 0:
        messages = [create_questions(description, num_questions=num_questions)]
    elif num_keywords > 0:
        messages = [create_keywords(description, num_keywords=num_keywords)]

    results = erniebot_chat(
        messages, api_type=args.api_type, access_token=args.access_token, temperature=temperature
    )["result"]
    return results


if __name__ == "__main__":
    erniebot.api_type = args.api_type
    erniebot.access_token = args.access_token

    embeddings = OpenAIEmbeddings(deployment="text-embedding-ada")
    faiss = FAISS.load_local("城市管理执法办法", embeddings)

    list_data = read_data(args.summarization_path)
    doc = list_data[0]
    tool_descriptions = generate_candidate_prompts(doc["abstract"], number_of_prompts=args.number_of_prompts)
    tool_descriptions = list(set(tool_descriptions))
    print(tool_descriptions)

    questions = generate_candidate_questions(doc["abstract"], num_questions=args.num_questions).split("\n")

    if args.num_keywords > 0:
        keywords = generate_candidate_questions(doc["abstract"], num_keywords=args.num_keywords).split("\n")
        questions += keywords

    prompts = tool_descriptions
    prompt_results = {prompt: {"correct": 0.0, "total": 0.0} for prompt in prompts}

    # Initialize the table
    table = PrettyTable()
    table_field_names = ["Prompt"] + [
        f"question {i+1}-{j+1}" for j, prompt in enumerate(questions) for i in range(questions.count(prompt))
    ]
    table.field_names = table_field_names

    # Wrap the text in the "Prompt" column
    table.max_width["Prompt"] = 100

    llm = ERNIEBot(model="ernie-bot")
    best_prompt = None
    best_percentage = 0.0
    for i, tool_description in enumerate(tool_descriptions):
        openai_city_management = OpenAISearchTool(
            name="city_administrative_law_enforcement",
            description=tool_description,
            db=faiss,
            threshold=0.1,
        )
        row = [tool_description]
        resps = []
        for query in questions:
            agent = PromptAgent(memory=WholeMemory(), llm=llm, tools=[openai_city_management])
            response = asyncio.run(agent.async_run(query))
            resps.append(response)
            if response is True:
                prompt_results[tool_description]["correct"] += 1
                row.append("✅")
            else:
                row.append("❌")
            prompt_results[tool_description]["total"] += 1
        table.add_row(row)

    print(f"生成的问题如下：{questions}")
    print(table)
    for i, prompt in enumerate(prompts):
        correct = prompt_results[prompt]["correct"]
        total = prompt_results[prompt]["total"]
        percentage = (correct / total) * 100
        print(f"Prompt {i+1} got {percentage:.2f}% correct.")
        if percentage > best_percentage:
            best_percentage = percentage
            best_prompt = tool_description

    print(f"The best prompt was '{best_prompt}' with a correctness of {best_percentage:.2f}%.")
