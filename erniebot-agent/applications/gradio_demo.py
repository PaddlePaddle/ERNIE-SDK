import argparse
import os

from erniebot_agent.agents import FunctionAgent
from erniebot_agent.chat_models import ERNIEBot
from erniebot_agent.tools import RemoteToolkit

def parse_args():
    parser = argparse.ArgumentParser(prog="erniebot-RPG")
    parser.add_argument("--access-token", type=str, default=None, help="Access token to use.")
    parser.add_argument("--model", type=str, default="ernie-3.5", help="Model name")
    return parser.parse_args()

if __name__ == '__main__':
    os.environ['EB_AGENT_LOGGING_LEVEL'] = 'info'
    args = parse_args()

    if os.getenv('EB_AGENT_ACCESS_TOKEN') is None and args.access_token is None:
        raise RuntimeError("Please set EB_AGENT_ACCESS_TOKEN in environment variables"
                           "or parse it in command line by --access-token.")
    
    if args.access_token is not None:
        access_token = args.access_token
    elif os.getenv('EB_AGENT_ACCESS_TOKEN') is not None:
        access_token = os.getenv('EB_AGENT_ACCESS_TOKEN')

    llm = ERNIEBot(
        model=args.model,
        api_type="aistudio",
        access_token=access_token,
        enable_multi_step_tool_call=True
    )
    tool = RemoteToolkit.from_aistudio("texttospeech").get_tools()[0]
    agent = FunctionAgent(
        llm=llm,
        tools=[tool],
    )
    agent.launch_gradio_demo()