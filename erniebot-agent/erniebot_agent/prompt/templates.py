# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from erniebot_agent.prompt import PromptTemplate

predefined_prompt_templates_list = [
    PromptTemplate(
        name="question-answering",
        template="Given the context please answer the question.\
             Context: {{documents|join(', ')}}; Question: "  # {join(documents)};
        "{{query}}; Answer:",
    ),
    PromptTemplate(
        name="question-answering-per-document",
        template="Given the context please answer the question. Context: {{documents}}; Question: "
        "{{query}}; Answer:",
    ),
    PromptTemplate(
        name="question-answering-with-document-scores",
        template="Answer the following question using the paragraphs below as sources. "
        "An answer should be short, a few words at most.\n"
        "Paragraphs:\n{{documents}}\n"
        "Question: {{query}}\n\n"
        "Instructions: Consider all the paragraphs above and their corresponding scores to generate "
        "the answer. While a single paragraph may have a high score, it's important to consider all "
        "paragraphs for the same answer candidate to answer accurately.\n\n"
        "After having considered all possibilities, the final answer is:\n",
    ),
    PromptTemplate(
        name="question-generation",
        template="Given the context please generate a question. Context: {{documents}}; Question:",
    ),
    PromptTemplate(
        name="conditioned-question-generation",
        template="Please come up with a question for the given context and the answer. "
        "Context: {{documents}}; Answer: {{answers}}; Question:",
    ),
    PromptTemplate(name="summarization", template="Summarize this document: {{documents}} Summary:"),
    PromptTemplate(
        name="question-answering-check",
        template="Does the following context contain the answer to the question? "
        "Context: {{documents}}; Question: {{query}}; Please answer yes or no! Answer:",
    ),
    PromptTemplate(
        name="sentiment-analysis",
        template="Please give a sentiment for this context. Answer with positive, "
        "negative or neutral. Context: {{documents}}; Answer:",
    ),
    PromptTemplate(
        name="multiple-choice-question-answering",
        template="Question:{{query}} ; Choose the most suitable option to answer the above question. "
        "Options: {{options}}; Answer:",
    ),
    PromptTemplate(
        name="topic-classification",
        template="Categories: {{options}}; What category best describes: {{documents}}; Answer:",
    ),
    PromptTemplate(
        name="language-detection",
        template="Detect the language in the following context and answer with the "
        "name of the language. Context: {{documents}}; Answer:",
    ),
    PromptTemplate(
        name="translation",
        template="Translate the following context to {{target_language}}.\
                 Context: {{documents}}; Translation:",
    ),
    PromptTemplate(
        name="zero-shot-react",
        template="You are a helpful and knowledgeable agent.\
            To achieve your goal of answering complex questions "
        "correctly, you have access to the following tools:\n\n"
        "{{tool_names_with_descriptions}}\n\n"
        "To answer questions, you'll need to go through \
            multiple steps involving step-by-step thinking and "
        "selecting appropriate tools and their inputs; tools will\
             respond with observations. When you are ready "
        "for a final answer, respond with the `Final Answer:`\n\n"
        "Use the following format:\n\n"
        "Question: the question to be answered\n"
        "Thought: Reason if you have the final answer. If yes, answer the question. If not, \
            find out the missing information needed to answer it.\n"
        "Tool: pick one of {{tool_names}} \n"
        "Tool Input: the input for the tool\n"
        "Observation: the tool will respond with the result\n"
        "...\n"
        "Final Answer: the final answer to the question, make it short (1-5 words)\n\n"
        "Thought, Tool, Tool Input, and Observation steps can be repeated multiple times, \
             but sometimes we can find an answer in the first pass\n"
        "---\n\n"
        "Question: {{query}}\n"
        "Thought: Let's think step-by-step, I first need to ",
    ),
    PromptTemplate(
        name="conversational-summary",
        template="Condense the following chat transcript by shortening and summarizing the \
             content without losing important information:\n{chat_transcript}\nCondensed Transcript:",
    ),
]
# todo: Add question answering with references template


def get_predefined_prompt_templates():
    """return a dictionary of predefined prompt templates"""
    predefined_prompt_templates = {}
    for template in predefined_prompt_templates_list:
        predefined_prompt_templates[template.name] = template

    return predefined_prompt_templates
