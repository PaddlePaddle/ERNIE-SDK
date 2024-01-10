import argparse
import asyncio
import hashlib
import os

import gradio as gr
from editor_actor_agent import EditorActorAgent
from langchain.embeddings.openai import OpenAIEmbeddings
from polish_agent import PolishAgent
from ranking_agent import RankingAgent
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
from tools.utils import FaissSearch, ReportCallbackHandler, build_index, setup_logging

from erniebot_agent.chat_models import ERNIEBot
from erniebot_agent.extensions.langchain.embeddings import ErnieEmbeddings
from erniebot_agent.memory import SystemMessage
from erniebot_agent.retrieval import BaizhongSearch

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
    "--index_name_citation", type=str, default="", help="The name of the citation base(faiss)"
)

parser.add_argument("--num_research_agent", type=int, default=2, help="The number of research agent")
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
parser.add_argument("--save_path", type=str, default="./output/erniebot", help="The report save path")
parser.add_argument("--server_name", type=str, default="0.0.0.0", help="the host of server")
parser.add_argument("--server_port", type=int, default=8878, help="the port of server")
parser.add_argument("--log_path", type=str, default="log.txt", help="Log file name")
args = parser.parse_args()
os.environ["api_type"] = args.api_type
access_token = os.environ.get("EB_AGENT_ACCESS_TOKEN", None)
# os.environ["EB_AGENT_LOGGING_FILE"] = args.log_path
# sh = logging.StreamHandler()
# logging.basicConfig(filename=args.log_path, level=logging.INFO)
logger = setup_logging(args.log_path)


def get_logs(path=args.log_path):
    file = open(path, "r")
    content = file.read()
    return content


def get_retrievers():
    if args.embedding_type == "openai_embedding":
        embeddings = OpenAIEmbeddings(deployment="text-embedding-ada")
        paper_db = build_index(faiss_name=args.index_name_full_text, embeddings=embeddings)
        abstract_db = build_index(faiss_name=args.index_name_abstract, embeddings=embeddings)
        abstract_search = FaissSearch(abstract_db, embeddings=embeddings)
        retriever_search = FaissSearch(paper_db, embeddings=embeddings)
    elif args.embedding_type == "ernie_embedding":
        embeddings = ErnieEmbeddings(aistudio_access_token=access_token)
        paper_db = build_index(faiss_name=args.index_name_full_text, embeddings=embeddings)
        abstract_db = build_index(faiss_name=args.index_name_abstract, embeddings=embeddings)
        abstract_search = FaissSearch(abstract_db, embeddings=embeddings)
        retriever_search = FaissSearch(paper_db, embeddings=embeddings)
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


def get_agents(retriever_sets, tool_sets, llm, llm_long, dir_path, target_path):
    research_actor = []
    for i in range(args.num_research_agent):
        agents_name = "agent_" + str(i)
        research_agent = ResearchAgent(
            name=agents_name,
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
            callbacks=ReportCallbackHandler(logger=logger),
        )
        research_actor.append(research_agent)
    editor_actor = EditorActorAgent(
        name="editor", llm=llm, llm_long=llm_long, callbacks=ReportCallbackHandler(logger=logger)
    )
    reviser_actor = ReviserActorAgent(
        name="reviser", llm=llm, llm_long=llm_long, callbacks=ReportCallbackHandler(logger=logger)
    )
    ranker_actor = RankingAgent(
        name="ranker",
        ranking_tool=tool_sets["ranking"],
        llm=llm,
        llm_long=llm_long,
        callbacks=ReportCallbackHandler(logger=logger),
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
        callbacks=ReportCallbackHandler(logger=logger),
    )
    return {
        "research_actor": research_actor,
        "editor_actor": editor_actor,
        "reviser_actor": reviser_actor,
        "ranker_actor": ranker_actor,
        "polish_actor": polish_actor,
    }


def generate_report(query, history=[]):
    dir_path = f"{args.save_path}/{hashlib.sha1(query.encode()).hexdigest()}"
    os.makedirs(dir_path, exist_ok=True)
    target_path = f"{args.save_path}/{hashlib.sha1(query.encode()).hexdigest()}/revised"
    os.makedirs(target_path, exist_ok=True)
    llm = ERNIEBot(model="ernie-4.0")
    llm_long = ERNIEBot(model="ernie-longtext")
    retriever_sets = get_retrievers()
    tool_sets = get_tools(llm, llm_long)
    agent_sets = get_agents(retriever_sets, tool_sets, llm, llm_long, dir_path, target_path)
    team_actor = ResearchTeam(**agent_sets, use_reflection=True)
    report, path = asyncio.run(team_actor.run(query, args.iterations))
    return report, path


def launch_ui():
    with gr.Blocks(title="报告生成小助手", theme=gr.themes.Base()) as demo:
        gr.HTML("""<h1 align="center">generation report小助手</h1>""")
        with gr.Row():
            with gr.Column():
                gr.Dropdown(
                    choices=[
                        "research_agent",
                        "editor_agent",
                        "ranking_agent",
                        "reviser_agent",
                        "user_agent",
                    ],
                    multiselect=True,
                    label="agents",
                    info="",
                )
        report = gr.Markdown(label="生成的report")
        report_url = gr.File(label="原文下载链接")
        with gr.Row():
            with gr.Column():
                query_textbox = gr.Textbox(placeholder="写一份关于机器学习发展的报告")
                gr.Examples(
                    [["写一份有关大模型技术发展的报告"], ["写一份数字经济发展的报告"], ["写一份关于机器学习发展的报告"]],
                    inputs=[query_textbox],
                    outputs=[query_textbox],
                    label="示例输入",
                )
            with gr.Row():
                submit = gr.Button("🚀 提交", variant="primary", scale=1)
                clear = gr.Button("清除", variant="primary", scale=1)
            submit.click(generate_report, inputs=[query_textbox], outputs=[report, report_url])
            clear.click(lambda _: ([None, None]), outputs=[report, report_url])
        recording = gr.Textbox(label="历史记录", max_lines=10)
        with gr.Row():
            clear_recoding = gr.Button(value="记录清除")
            submit_recoding = gr.Button(value="记录更新")
        submit_recoding.click(get_logs, inputs=[], outputs=[recording])
        clear_recoding.click(lambda _: ([[None, None]]), outputs=[recording])
    demo.launch(server_name=args.server_name, server_port=args.server_port)


if "__main__" == __name__:
    launch_ui()
