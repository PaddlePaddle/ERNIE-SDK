# ERNIE Researcher 

ERNIE Researcher是一个自主智能体（Autonomous Agent），旨在对各种任务进行全面的在线研究。

Agent可以制作详细、真实和公正的研究报告，并提供专注于相关资源、大纲和经验教训的定制选项。受最近的[Plan-and-Solve](https://arxiv.org/abs/2305.04091) 和 当前最流行的[RAG](https://arxiv.org/abs/2005.11401)技术的启发， ERNIE Researcher解决了速度、确定性和可靠性问题，通过多Agent协作和并行工作提供了更稳定的性能和更高的速度。

## 为什么需要ERNIE Researcher？

+ 手动研究任务形成客观结论可能需要时间，有时需要数周才能找到正确的资源和信息。
+ 目前的LLM是根据过去和过时的信息进行训练的，产生幻觉的风险很大，这使得产生的报告几乎与研究任务无关。
+ 支持网络搜索的解决方案只考虑有限的资源和内容，在某些情况下会导致肤浅的结论或有偏见的答案。
+ LLM生成的报告一般没有做段落级/句子级别的文献来源引用，生成的内容无法进行追踪和验证。

## 架构

主要思想是运行“planner”和“execution” agents，而planner生成问题进行研究，execution agents根据每个生成的研究问题寻求最相关的信息。最后，planner 过滤并汇总所有相关信息，并创建一份研究报告。

Agents利用ernie-4.0和ernie-longtext来完成研究任务， ernie-4.0主要用于做决策和规划，ernie-longtext主要用于撰写报告。


<div align="center">
    <img src="https://github.com/PaddlePaddle/ERNIE-Bot-SDK/assets/12107462/a265ff50-eab1-41bb-9291-6f4b6c6597cf" width="500px">
</div>

具体地：

+ 基于研究查询或任务创建特定领域的Agent。
+ 根据现有知识库的内容生成一组多样性的研究问题，这些问题共同形成对任何给定任务的客观意见。
+ 对于每个研究问题，从知识库中选择与给定问题相关的信息。
+ 过滤和汇总所有的信息来源，并生成最终的研究报告。
+ 多个报告并行生成，并保持一定的多样性。
+ 对多个报告进行质量评估和排序，克服伪随机性，并选择最优的报告。
+ 对报告进行修订和编辑。


## 快速开始

**注意** 生成一次报告需要花费10min以上，并且会消耗大量的Tokens。

> 第一步：下载项目源代码

```
git clone https://github.com/PaddlePaddle/ERNIE-Bot-SDK.git
cd ernie-agent/applications/erniebot_researcher
```

> 第二步：安装依赖

```
pip install -r requirements.txt
```

> 第三步：运行

```
python ui.py --access_token <aistudio-access-token>
```



## :heart: Acknowledge
我们借鉴了 Assaf Elovic [GPT Researcher](https://github.com/assafelovic/gpt-researcher) 优秀的框架设计，在此对[GPT Researcher](https://github.com/assafelovic/gpt-researcher)作者及其开源社区表示感谢。

We learn form the excellent framework design of Assaf Elovic [GPT Researcher](https://github.com/assafelovic/gpt-researcher), and we would like to express our thanks to the authors of GPT Researcher and their open source community.
