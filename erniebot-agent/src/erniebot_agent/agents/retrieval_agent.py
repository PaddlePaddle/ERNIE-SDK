from typing import List, Optional, Sequence

from erniebot_agent.agents.agent import Agent
from erniebot_agent.agents.schema import AgentResponse, AgentStep
from erniebot_agent.file import File
from erniebot_agent.memory.messages import HumanMessage, Message
from erniebot_agent.prompt import PromptTemplate

GRAPH_PROMPT = """id: Unique id of the query.
dependancies: List of sub questions that need to be answered before we can ask the question. \
    Use a subquery when anything may be unknown, and we need to ask multiple questions to get the answer. \
    Dependences must only be other queries.
question: Question we are asking using a question answer system, if we are asking multiple questions, \
    this question is asked by also providing the answers to the sub questions
example:
##
query: What is the difference in populations of Canada and the Jason's home country?
query_graph: {'query_graph': [{'dependancies': [],
                    'id': 1,
                    'question': "Identify Jason's home country"},
                    {'dependancies': [],
                    'id': 2,
                    'question': 'Find the population of Canada'},
                    {'dependancies': [1],
                    'id': 3,
                    'question': "Find the population of Jason's home country"},
                    {'dependancies': [2, 3],
                    'id': 4,
                    'question': 'Calculate the difference in populations between '
                                "Canada and Jason's home country"}]}
##
query: {{query}}
query_graph:
"""


class DAGRetrievalAgent(Agent):
    def __init__(
        self,
        knowledge_base,
        top_k: int = 2,
        threshold: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.query_transform = PromptTemplate(GRAPH_PROMPT, input_variables=["query"])

    async def _run(self, prompt: str, files: Optional[Sequence[File]] = None) -> AgentResponse:
        steps_taken: List[AgentStep] = []

        steps_input = HumanMessage(content=self.query_transform.format(query=prompt))
        chat_history: List[Message] = [steps_input]
        # Query planning
        llm_resp = await self.run_llm(
            messages=[steps_input],
        )
        output_message = llm_resp.message
        chat_history.append(output_message)
        self.memory.add_message(chat_history[0])
        self.memory.add_message(chat_history[-1])
        response = self._create_finished_response(chat_history, steps_taken)
        return response

    def _create_finished_response(
        self,
        chat_history: List[Message],
        steps: List[AgentStep],
    ) -> AgentResponse:
        last_message = chat_history[-1]
        return AgentResponse(
            text=last_message.content,
            chat_history=chat_history,
            steps=steps,
            status="FINISHED",
        )
