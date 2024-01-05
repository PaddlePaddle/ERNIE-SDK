import argparse
import asyncio
import hashlib
import os
import time

from editor_actor_agent import EditorActorAgent
from langchain.embeddings.openai import OpenAIEmbeddings
from ranking_agent import RankingAgent
from render_agent import RenderAgent
from research_agent import ResearchAgent
from research_team import ResearchTeam
from reviser_actor_agent import ReviserActorAgent
from tools.intent_detection_tool import IntentDetectionTool
from tools.outline_generation_tool import OutlineGenerationTool
from tools.ranking_tool import TextRankingTool
from tools.report_writing_tool import ReportWritingTool
from tools.semantic_citation_tool import SemanticCitationTool
from tools.summarization_tool import TextSummarizationTool
from tools.task_planning_tool import TaskPlanningTool
from tools.utils import FaissSearch, build_index

from erniebot_agent.chat_models import ERNIEBot
from erniebot_agent.extensions.langchain.embeddings import ErnieEmbeddings
from erniebot_agent.memory import SystemMessage
from erniebot_agent.retrieval import BaizhongSearch

parser = argparse.ArgumentParser()
parser.add_argument("--api_type", type=str, default="aistudio")

parser.add_argument("--knowledge_base_name_paper", type=str, default="", help="")
parser.add_argument("--knowledge_base_name_abstract", type=str, default="", help="")
parser.add_argument("--knowledge_base_id_paper", type=str, default="", help="")
parser.add_argument("--knowledge_base_id_abstract", type=str, default="", help="")

parser.add_argument("--faiss_name_paper", type=str, default="", help="")
parser.add_argument("--faiss_name_abstract", type=str, default="", help="")
parser.add_argument("--faiss_name_citation", type=str, default="", help="")

parser.add_argument("--num_research_agent", type=int, default=2, help="The number of research agent")
parser.add_argument("--iterations", type=int, default=4, help="")
parser.add_argument(
    "--report_type",
    type=str,
    default="research_report",
    help="['research_report','resource_report','outline_report']",
)
parser.add_argument(
    "--embedding_type",
    type=str,
    default="open_embedding",
    help="['open_embedding','baizhong','ernie_embedding']",
)

parser.add_argument("--server_name", type=str, default="0.0.0.0")
parser.add_argument("--server_port", type=int, default=8878)
args = parser.parse_args()
os.environ["api_type"] = args.api_type
access_token = os.environ.get("EB_AGENT_ACCESS_TOKEN", None)


def get_retrievers():
    if args.embedding_type == "open_embedding":
        embeddings = OpenAIEmbeddings(deployment="text-embedding-ada")
        paper_db = build_index(faiss_name=args.faiss_name_paper, embeddings=embeddings)
        abstract_db = build_index(faiss_name=args.faiss_name_abstract, embeddings=embeddings)
        abstract_search = FaissSearch(abstract_db, embeddings=embeddings)
        retriever_search = FaissSearch(paper_db, embeddings=embeddings)
    elif args.embedding_type == "ernie_embedding":
        embeddings = ErnieEmbeddings(aistudio_access_token=access_token)
        paper_db = build_index(faiss_name=args.faiss_name_paper, embeddings=embeddings)
        abstract_db = build_index(faiss_name=args.faiss_name_abstract, embeddings=embeddings)
        abstract_search = FaissSearch(abstract_db, embeddings=embeddings)
        retriever_search = FaissSearch(paper_db, embeddings=embeddings)
    elif args.embedding_type == "baizhong":
        embeddings = ErnieEmbeddings(aistudio_access_token=access_token)
        retriever_search = BaizhongSearch(
            access_token=access_token,
            knowledge_base_name=args.knowledge_base_name_paper,
            knowledge_base_id=args.knowledge_base_id_paper,
        )
        abstract_search = BaizhongSearch(
            access_token=access_token,
            knowledge_base_name=args.knowledge_base_name_abstract,
            knowledge_base_id=args.knowledge_base_id_abstract,
        )
    return {"full_text": retriever_search, "abstract": abstract_search, "embeddings": embeddings}


def get_tools(llm, llm_long):
    intent_detection_tool = IntentDetectionTool(llm=llm)
    outline_generation_tool = OutlineGenerationTool(llm=llm)
    ranking_tool = TextRankingTool(llm=llm, llm_long=llm_long)
    report_writing_tool = ReportWritingTool(llm=llm, llm_long=llm_long)
    summarization_tool = TextSummarizationTool()
    task_planning_tool = TaskPlanningTool(llm=llm)
    semantic_citation_tool = SemanticCitationTool(llm=llm, theta_min=0.7)

    return {
        "intent_detection": intent_detection_tool,
        "outline": outline_generation_tool,
        "ranking": ranking_tool,
        "report_writing": report_writing_tool,
        "text_summarization": summarization_tool,
        "task_planning": task_planning_tool,
        "semantic_citation": semantic_citation_tool,
    }


def get_agents(retriever_sets, tool_sets, llm, llm_long):
    dir_path = f"./outputs/erniebot/{hashlib.sha1(query.encode()).hexdigest()}"
    os.makedirs(dir_path, exist_ok=True)

    target_path = f"./outputs/erniebot/{hashlib.sha1(query.encode()).hexdigest()}/revised"
    os.makedirs(target_path, exist_ok=True)
    research_actor = []
    for i in range(args.num_research_agent):
        agents_name = "agent_" + str(i)
        research_agent = ResearchAgent(
            name="generate_report",
            agent_name=agents_name,
            system_message=SystemMessage("你是一个报告生成助手。你可以根据用户的指定内容生成一份报告手稿"),
            dir_path=dir_path,
            report_type=args.report_type,
            retriever_abstract_tool=retriever_sets["abstract"],
            retriever_tool=retriever_sets["full_text"],
            intent_detection_tool=tool_sets["intent_detection"],
            task_planning_tool=tool_sets["task_planning"],
            report_writing_tool=tool_sets["report_writing"],
            outline_tool=tool_sets["outline"],
            summarize_tool=tool_sets["text_summarization"],
            llm=llm,
        )
        research_actor.append(research_agent)
    editor_actor = EditorActorAgent(name="editor", llm=llm, llm_long=llm_long)
    reviser_actor = ReviserActorAgent(name="reviser", llm=llm, llm_long=llm_long)
    render_actor = RenderAgent(
        name="render",
        llm=llm,
        llm_long=llm_long,
        citation_tool=tool_sets["semantic_citation"],
        faiss_name_citation=args.faiss_name_citation,
        embeddings=retriever_sets["embeddings"],
        dir_path=target_path,
        report_type=args.report_type,
    )
    ranker_actor = RankingAgent(
        llm=llm,
        llm_long=llm_long,
        name="ranker",
        ranking_tool=tool_sets["ranking"],
    )
    return {
        "research_agents": research_actor,
        "editor": editor_actor,
        "reviser": reviser_actor,
        "ranker": ranker_actor,
        "render": render_actor,
    }


def main(query):
    llm_long = ERNIEBot(model="ernie-longtext")
    llm = ERNIEBot(model="ernie-4.0")

    retriever_sets = get_retrievers()
    tool_sets = get_tools(llm, llm_long)
    agent_sets = get_agents(retriever_sets, tool_sets, llm, llm_long)
    research_team = ResearchTeam(
        research_actor=agent_sets["research_agents"],
        ranker_actor=agent_sets["ranker"],
        editor_actor=agent_sets["editor"],
        reviser_actor=agent_sets["reviser"],
        render_actor=agent_sets["render"],
    )

    report, file_path = asyncio.run(research_team.run(query))
    print(file_path)
    print(report)


if "__main__" == __name__:
    query = "写一份有关大模型技术发展的报告"
    start_time = time.time()
    main(query)
    end_time = time.time()
    print("Took time: {}".format(end_time - start_time))
