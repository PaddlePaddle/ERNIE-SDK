export EB_MAX_RETRIES=100
# export EB_BASE_URL=http://10.154.81.14:8868/ernie-foundry/v1

export EB_API_TYPE="aistudio"
export EB_ACCESS_TOKEN="7d94e511182b340758dd9835de044a7865b736b2"
export EB_ACCESS_TOKEN="1dc43e5843cfb51b7b41ba766aff2372cf2f3ccb"
export EB_ACCESS_TOKEN="9ca7b061ece73343c9ea457f4c64f27d3fe15221"

# token = "9ca7b061ece73343c9ea457f4c64f27d3fe15221"
export EB_LOGGING_LEVEL="info"

# export AISTUDIO_HUB_BASE_URL=http://sandbox-aistudio-hub.baidu.com

base_dir=/Users/wujingjing05/projects/yiyan/ERNIE-Bot-SDK-mkdoc

export PYTHONPATH=${base_dir}/erniebot-agent/src:$PYTHONPATH
export PYTHONPATH=${base_dir}/erniebot/src:$PYTHONPATH

# python tool_simple_with_eb.py
python tool_a.py


