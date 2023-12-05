import json
from typing import List, Optional

from erniebot_agent.agents.functional_agent import FunctionalAgent
from erniebot_agent.agents.schema import AgentAction, AgentFile
from erniebot_agent.messages import HumanMessage, Message
from erniebot_agent.retrieval.baizhong_search import BaizhongSearch

_MAX_RETRY_STEPS = 5


def create_search_intent(chunk: str, query: str) -> str:
    context = f"{chunk}，请判断上面的关键信息和{query}是否相关?"
    return (
        context
        + """按照下面的格式输出：
            如果相关，则回复：{"msg":"相关"}，如果不相关，则回复：{"msg":"不相关"}。回复："""
    )


class FunctionalAgentWithRetrieval(FunctionalAgent):
    def __init__(self, knowledge_base: BaizhongSearch, top_k: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.knowledge_base = knowledge_base
        self.top_k = kwargs.get("top_k", 3)

    async def _async_step(
        self,
        step_input,
        chat_history: List[Message],
        actions: List[AgentAction],
        files: List[AgentFile],
    ) -> Optional[Message]:
        if self.knowledge_base is not None:
            results = await self._post_process(step_input, chat_history, actions, files)

            if results["msg"] == "相关":
                step_input = HumanMessage(
                    content="背景：" + results["retrieval_results"] + "请根据上面的背景信息回答下面的问题：" + step_input.content
                )
                # Currently, _async_step do not support the retrieval-based agent
                # return await super()._async_step(step_input, chat_history, actions, files)
                chat_history.append(step_input)
                llm_resp = await self._async_run_llm(
                    messages=chat_history,
                    functions=None,
                    system=self.system_message.content if self.system_message is not None else None,
                )
                output_message = llm_resp.message
                chat_history.append(output_message)
                return None
            elif results["msg"] == "不相关":
                return await super()._async_step(step_input, chat_history, actions, files)
            else:
                raise ValueError("No answer found in the knowledge base")
        else:
            return await super()._async_step(step_input, chat_history, actions, files)

    def _pre_process(self, step_input):
        # TODO: add pre-process logic
        raise NotImplementedError

    async def _post_process(
        self,
        step_input,
        chat_history: List[Message],
        actions: List[AgentAction],
        files: List[AgentFile],
    ):
        documents = self.knowledge_base.search(step_input.content, top_k=self.top_k, filters=None)
        retrieval_results = ""
        for index, item in enumerate(documents):
            retrieval_results += f"第{index}个段落：{item['content_se']}\n"

        messages = [HumanMessage(content=create_search_intent(retrieval_results, step_input))]
        response = await self._async_run_llm(messages)
        message = response.message
        results = self._parse_results(message.content)
        request_count = 0
        while isinstance(results, str):
            response = await self._async_run_llm(messages)
            message = response.message
            results = self._parse_results(message.content)
            request_count += 1
            if request_count > _MAX_RETRY_STEPS:
                break
        return {"msg": results["msg"], "retrieval_results": retrieval_results}

    def _parse_results(self, results):
        left_index = results.find("{")
        right_index = results.rfind("}")
        if left_index == -1 or right_index == -1:
            return results
        try:
            return json.loads(results[left_index : right_index + 1])
        except Exception:
            return results
