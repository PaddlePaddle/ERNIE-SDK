import argparse
import os

os.environ["EB_AGENT_LOGGING_LEVEL"] = "INFO"

from init_vector_db import init_db
from langchain.vectorstores import FAISS
from sklearn.metrics.pairwise import cosine_similarity

from erniebot_agent.agents.function_agent_with_retrieval import (
    FunctionAgentWithRetrieval,
)
from erniebot_agent.chat_models import ERNIEBot
from erniebot_agent.extensions.langchain.embeddings import ErnieEmbeddings
from erniebot_agent.memory import SystemMessage

parser = argparse.ArgumentParser()
parser.add_argument("--init", type=bool, default=False)
parser.add_argument("--access-token", type=str, help="access token for erniebot-agent")
args = parser.parse_args()

if args.access_token:
    os.environ["EB_AGENT_ACCESS_TOKEN"] = args.access_token

embeddings = ErnieEmbeddings(aistudio_access_token=os.environ["EB_AGENT_ACCESS_TOKEN"], chunk_size=16)


class FaissSearch:
    def __init__(self, db, embeddings, module_code_db):
        self.db = db
        self.module_code_db = module_code_db
        self.embeddings = embeddings

    def search(self, query: str, top_k: int = 2):
        # 搜索时，同时召回最相关的两个文档片段以及最相关的一个代码示例
        docs = self.db.similarity_search(query, top_k)
        para_result = self.embeddings.embed_documents([i.page_content for i in docs])
        query_result = self.embeddings.embed_query(query)
        similarities = cosine_similarity([query_result], para_result).reshape((-1,))
        retrieval_results = []
        # make sure 'raw_text' in doc.metadata
        for index, doc in enumerate(docs):
            if "Header 1" in doc.metadata:
                retrieval_results.append(
                    {
                        "content": doc.metadata["raw_text"],
                        "score": similarities[index],
                        "title": doc.metadata["Header 1"],
                    }
                )
            else:
                retrieval_results.append(
                    {"content": doc.metadata["raw_text"], "score": similarities[index], "title": ""}
                )
        # module_code_db 用于相关代码的召回
        code = self.module_code_db.similarity_search(query, 1)[0]
        # make sure 'ipynb' in code.metadata
        retrieval_results.append({"content": code.metadata["ipynb"], "score": 1, "title": code.page_content})

        return retrieval_results


def load_agent():
    faiss_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faiss_index")
    faiss_name_module = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faiss_index_module")
    if args.init:
        init_db(faiss_name, faiss_name_module, embeddings)
    try:
        db = FAISS.load_local(faiss_name, embeddings)
        module_code_db = FAISS.load_local(faiss_name_module, embeddings)
    except RuntimeError as e:
        raise RuntimeError(f"Make sure you have initialized the database first.\n {e}")

    llm = ERNIEBot(model="ernie-3.5")
    faiss_search = FaissSearch(db=db, embeddings=embeddings, module_code_db=module_code_db)
    agent = FunctionAgentWithRetrieval(
        llm=llm,
        tools=[],
        knowledge_base=faiss_search,
        threshold=0,
        system_message=SystemMessage(
            "你是ERNIEBot Agent的小助手，用于解决用户关于EB-Agent的问题，涉及File, Memory, Message, Agent, ChatModels等模块。"
            "请你严格按照搜索到的内容回答，不要自己生成相关代码。如果询问与ERNIEBot Agent无关的问题，请直接回答“我只能回答EB—Agent相关问题”"
        ),
        top_k=2,
        token_limit=5000,
    )
    return agent


async def main(agent):
    # response = await agent.run('怎么从aistudio创建远程tool？')
    # response = await agent.run('如何创建一个LocalTool？')
    response = await agent.run("如何创建一个agent")
    print(response.text)


if __name__ == "__main__":
    agent = load_agent()
    # asyncio.run(main(agent))
    agent.launch_gradio_demo()
