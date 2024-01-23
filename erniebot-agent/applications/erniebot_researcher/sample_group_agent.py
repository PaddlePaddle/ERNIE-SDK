import argparse
import asyncio
import hashlib
import os
import time

from editor_actor_agent import EditorActorAgent
from group_agent import GroupChat, GroupChatManager
from langchain_openai import AzureOpenAIEmbeddings
from polish_agent import PolishAgent
from ranking_agent import RankingAgent
from research_agent import ResearchAgent
from reviser_actor_agent import ReviserActorAgent
from tools.intent_detection_tool import IntentDetectionTool
from tools.outline_generation_tool import OutlineGenerationTool
from tools.preprocessing import get_retriver_by_type
from tools.ranking_tool import TextRankingTool
from tools.report_writing_tool import ReportWritingTool
from tools.semantic_citation_tool import SemanticCitationTool
from tools.summarization_tool import TextSummarizationTool
from tools.task_planning_tool import TaskPlanningTool

from erniebot_agent.chat_models import ERNIEBot
from erniebot_agent.extensions.langchain.embeddings import ErnieEmbeddings
from erniebot_agent.memory import SystemMessage
from erniebot_agent.retrieval import BaizhongSearch

access_token = os.environ.get("EB_AGENT_ACCESS_TOKEN", None)
parser = argparse.ArgumentParser()
parser.add_argument("--api_type", type=str, default="aistudio")

parser.add_argument(
    "--knowledge_base_name_full_text",
    type=str,
    default="",
    help="The name of the full-text knowledge base(baizhong)",
)
parser.add_argument(
    "--knowledge_base_name_abstract", type=str, default="", help="The name of the abstract base(baizhong)"
)
parser.add_argument(
    "--knowledge_base_id_full_text",
    type=str,
    default="",
    help="The id of the full-text knowledge base(baizhong)",
)
parser.add_argument(
    "--knowledge_base_id_abstract", type=str, default="", help="The id of the abstract base(baizhong)"
)

parser.add_argument(
    "--index_name_full_text", type=str, default="", help="The name of the full-text knowledge base(faiss)"
)
parser.add_argument(
    "--index_name_abstract", type=str, default="", help="The name of the abstract base(faiss)"
)
parser.add_argument(
    "--index_name_citation", type=str, default="citation_index", help="The name of the citation base(faiss)"
)
parser.add_argument("--save_path", type=str, default="./output/erniebot", help="The report save path")
parser.add_argument("--iterations", type=int, default=4, help="Maximum number of corrections")
parser.add_argument(
    "--report_type",
    type=str,
    default="research_report",
    help="['research_report','resource_report','outline_report']",
)
parser.add_argument(
    "--embedding_type",
    type=str,
    default="openai_embedding",
    help="['openai_embedding','baizhong','ernie_embedding']",
)
parser.add_argument(
    "--use_frame",
    type=str,
    default="langchain",
    choices=["langchain", "llama_index"],
    help="['langchain','llama_index']",
)
args = parser.parse_args()


def get_retrievers(build_index_function, retrieval_tool):
    if args.embedding_type == "openai_embedding":
        embeddings = AzureOpenAIEmbeddings(azure_deployment="text-embedding-ada")
        paper_db = build_index_function(faiss_name=args.index_name_full_text, embeddings=embeddings)
        abstract_db = build_index_function(faiss_name=args.index_name_abstract, embeddings=embeddings)
        abstract_search = retrieval_tool(abstract_db, embeddings=embeddings)
        retriever_search = retrieval_tool(paper_db, embeddings=embeddings)
    elif args.embedding_type == "ernie_embedding":
        embeddings = ErnieEmbeddings(aistudio_access_token=access_token)
        paper_db = build_index_function(faiss_name=args.index_name_full_text, embeddings=embeddings)
        abstract_db = build_index_function(faiss_name=args.index_name_abstract, embeddings=embeddings)
        abstract_search = retrieval_tool(abstract_db, embeddings=embeddings)
        retriever_search = retrieval_tool(paper_db, embeddings=embeddings)
    elif args.embedding_type == "baizhong":
        embeddings = ErnieEmbeddings(aistudio_access_token=access_token)
        retriever_search = BaizhongSearch(
            access_token=access_token,
            knowledge_base_name=args.knowledge_base_name_full_text,
            knowledge_base_id=args.knowledge_base_id_full_text,
        )
        abstract_search = BaizhongSearch(
            access_token=access_token,
            knowledge_base_name=args.knowledge_base_name_abstract,
            knowledge_base_id=args.knowledge_base_id_abstract,
        )
    return {"full_text": retriever_search, "abstract": abstract_search, "embeddings": embeddings}


def get_agents(
    retriever_sets, tool_sets, llm, llm_long, dir_path, target_path, build_index_function, retrieval_tool
):
    research_actor = ResearchAgent(
        name="generate_report",
        system_message=SystemMessage("你是一个报告生成助手。你可以根据用户的指定内容生成一份报告手稿"),
        dir_path=dir_path,
        report_type=args.report_type,
        retriever_abstract_db=retriever_sets["abstract"],
        retriever_fulltext_db=retriever_sets["full_text"],
        intent_detection_tool=tool_sets["intent_detection"],
        task_planning_tool=tool_sets["task_planning"],
        report_writing_tool=tool_sets["report_writing"],
        outline_tool=tool_sets["outline"],
        summarize_tool=tool_sets["text_summarization"],
        llm=llm,
    )
    editor_actor = EditorActorAgent(name="editor", llm=llm, llm_long=llm_long)
    reviser_actor = ReviserActorAgent(name="reviser", llm=llm, llm_long=llm_long)
    ranker_actor = RankingAgent(
        name="ranker",
        ranking_tool=tool_sets["ranking"],
        llm=llm,
        llm_long=llm_long,
    )
    polish_actor = PolishAgent(
        name="polish",
        llm=llm,
        llm_long=llm_long,
        citation_index_name=args.index_name_citation,
        embeddings=retriever_sets["embeddings"],
        dir_path=target_path,
        report_type=args.report_type,
        citation_tool=tool_sets["semantic_citation"],
        build_index_function=build_index_function,
        search_tool=retrieval_tool,
    )
    return {
        "research_agents": research_actor,
        "editor": editor_actor,
        "reviser": reviser_actor,
        "ranker": ranker_actor,
        "polish": polish_actor,
    }


def get_tools(llm, llm_long):
    intent_detection_tool = IntentDetectionTool(llm=llm)
    outline_generation_tool = OutlineGenerationTool(llm=llm)
    ranking_tool = TextRankingTool(llm=llm, llm_long=llm_long)
    report_writing_tool = ReportWritingTool(llm=llm, llm_long=llm_long)
    summarization_tool = TextSummarizationTool()
    task_planning_tool = TaskPlanningTool(llm=llm)
    semantic_citation_tool = SemanticCitationTool()

    return {
        "intent_detection": intent_detection_tool,
        "outline": outline_generation_tool,
        "ranking": ranking_tool,
        "report_writing": report_writing_tool,
        "text_summarization": summarization_tool,
        "task_planning": task_planning_tool,
        "semantic_citation": semantic_citation_tool,
    }


def main(query):
    dir_path = f"{args.save_path}/{hashlib.sha1(query.encode()).hexdigest()}"
    os.makedirs(dir_path, exist_ok=True)
    target_path = f"{args.save_path}/{hashlib.sha1(query.encode()).hexdigest()}/revised"
    os.makedirs(target_path, exist_ok=True)
    llm_long = ERNIEBot(model="ernie-longtext")
    llm = ERNIEBot(model="ernie-4.0")
    build_index_function, retrieval_tool = get_retriver_by_type(args.use_frame)
    retriever_sets = get_retrievers(build_index_function, retrieval_tool)
    tool_sets = get_tools(llm, llm_long)
    agent_sets = get_agents(
        retriever_sets, tool_sets, llm, llm_long, dir_path, target_path, build_index_function, retrieval_tool
    )
    research_actor = agent_sets["research_agents"]
    report = asyncio.run(research_actor.run(query))
    report = {"report": report[0], "paragraphs": report[1]}
    agent_list = [
        agent_sets["research_agents"],
        agent_sets["ranker"],
        agent_sets["editor"],
        agent_sets["reviser"],
        agent_sets["polish"],
    ]
    group_chat = GroupChat(agent_list, max_round=5, llm=llm, llm_long=llm_long)
    group_manager = GroupChatManager(group_chat)
    report = asyncio.run(group_manager.run(query=query, report=report, speaker=research_actor))


if "__main__" == __name__:
    query = "写一份有关大模型技术发展的报告"
    start_time = time.time()
    main(query)
    end_time = time.time()
    print("Took time: {}".format(end_time - start_time))
