import asyncio

from erniebot_agent.agents.functional_agent import FunctionalAgent
from erniebot_agent.chat_models.erniebot import ERNIEBot
from erniebot_agent.file_io import get_file_manager
from erniebot_agent.memory.whole_memory import WholeMemory
from erniebot_agent.tools.base import RemoteToolkit


class CVToolkit:
    def __init__(self):
        API_URL = "<your-api-url>"
        self.toolkit = RemoteToolkit.from_url(API_URL, access_token="<your-access-token>")
        self.tools = self.toolkit.get_tools()


llm = ERNIEBot(model="ernie-bot", api_type="custom")
# llm = ERNIEBot(model="ernie-bot", api_type="aistudio", access_token="<your-access-token>")
toolkit = CVToolkit()
memory = WholeMemory()
file_manager = get_file_manager()
agent = FunctionalAgent(llm=llm, tools=toolkit.tools, memory=memory, file_manager=file_manager)


async def run_agent():
    seg_file = await file_manager.create_file_from_path(file_path="cityscapes_demo.png", file_type="local")
    clas_file = await file_manager.create_file_from_path(file_path="class_img.jpg", file_type="local")
    ocr_file = await file_manager.create_file_from_path(file_path="ch.png", file_type="local")
    agent_resp = await agent.async_run(f"这张图片中包含什么中文文字？{ocr_file.id}")  # 单张输入
    print("=" * 10 + "OCR返回结果" + "=" * 10 + "\n")
    print(agent_resp.text)
    print("\n" + "=" * 20 + "\n")
    agent_resp = await agent.async_run(
        f"请对第一张图片中的物体进行分类然后对第二张图片中的物体进行分割？{clas_file.id}，{seg_file.id}"
    )  # 多张输入
    print("=" * 10 + "分类+分割返回结果" + "=" * 10 + "\n")
    print(agent_resp.text)
    print("\n" + "=" * 20 + "\n")
    agent_resp = await agent.async_run("请对上一张分割后的结果进行分类")  # 上一轮指代
    print("=" * 10 + "上一轮图片分割" + "=" * 10 + "\n")
    print(agent_resp.text)
    print("\n" + "=" * 20 + "\n")
    agent_resp = await agent.async_run("我还想对第一张图片进行分类")  # 多轮指代
    print("=" * 10 + "OCR图片分类" + "=" * 10 + "\n")
    print(agent_resp.text)
    print("\n" + "=" * 20 + "\n")
    agent_resp = await agent.async_run(f"请帮我将这张中的汽车分割出来{seg_file.id}")
    print("=" * 10 + "分割返回结果" + "=" * 10 + "\n")
    print(agent_resp.text)
    print(agent_resp.annotations)
    print("\n" + "=" * 20 + "\n")


if __name__ == "__main__":
    asyncio.run(run_agent())
    # agent.launch_gradio_demo(server_name="0.0.0.0", server_port=8017, share=True) # gradio 还没加上文件上传等
