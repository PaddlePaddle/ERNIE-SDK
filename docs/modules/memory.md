# Memory 模块介绍
## 1. 模块简介
在建立一个Agent应用的过程中，由于LLM本身是无状态的，因此很重要的一点就是赋予Agent记忆能力。Agent的记忆能力主要可以分为长期记忆和短期记忆。

* 长期记忆通过文件/数据库的形式存储，是不会被遗忘的内容，每次判断需要相关知识就可以retrieval的方式，找到最相关的内容放入消息帮助LLM分析得到结果。
* 短期记忆则是直接在消息中体现，是LLM接触的一手信息，但是也受限于上下文窗口，更容易被遗忘。

这里我们简述在我们记录短期记忆的方式，即在Memory模块中存储消息，并通过消息裁剪，控制消息总条数不会超过上下文窗口。

在使用层面，Memory将传入Agent类中，用于记录多轮的消息，即Agent会在每一轮对话中和Memory有一次交互：即在LLM产生最终结论之后，将第一条HumanMessage和最后一条AIMessage加入到Memory中。


## 2. 核心类
在Memory类的内部，我们将其主要分为两个部分，即存放对消息处理逻辑的Memory以及存放消息的数据集结构Msg_manager。

### 2.1 类接口和关系介绍

#### Memory基类：

* 属性：msg_manager（管理消息的增删改查）
* 方法：
    * add_messages(self, messages): 批量地增加message。
    * add_message(self, message): 增加一条message。
    * get_messages(self): 获取Memory中的所有messages。
    * get_system_message(self): 获取Memory中的系统消息，Memory中有且仅有一条系统信息，可以传入LLM的* * system接口，用于建立LLM的特性。
    * clear_chat_history(self): 清除memory中所有message的历史。
* 关系：Memory的基类，关联到MessageManager类。


#### MessageManager类：用于存储message的数据结构，基础实现使用list存储，也可以替换成其他数据结构

* 属性：messages（存放message的列表）
* 方法：
    * add_messages(self, messages): 增加多条message。
    * add_message(self, message): 增加一条message。
    * system_message(self): 获取和存储系统消息。
    * pop_message(self, index): 在指定位置删除一条message。
    * clear_messages(self): 清空所有message。
    * update_last_message_token_count(self, token_count): 更新最后一条message的token_count。
    * retrieve_messages(self): 获取所有的message。
* 关系：
    * MessageManager类，传入到Memory类中。
    * 通过List存储Message类的对象。

## 3. 几种Memory的变体
为了更好地适应上下文窗口，我们支持了几种memory变体，后续也将从存储更多的语义等角度出发，增加更多的Memory类型。目前我们支持的Memory和功能描述如下：

| 支持的Memory名称 | 功能描述 | 代码链接
| :--: | :--: | :--: |
| WholeMemory| 支持存储所有的消息| [whole_memory.py](../../erniebot-agent/src/erniebot_agent/memory/whole_memory.py) |
| LimitTokensMemory| 根据消息中所占用token的数量，删除最前面的一些message| [limit_token_memory.py](../../erniebot-agent/src/erniebot_agent/memory/limit_token_memory.py) |
| SlidingWindowMemory| 通过滑窗的方式，限制消息的轮数，并支持保留前k轮messages| [sliding_window_memory.py](../../erniebot-agent/src/erniebot_agent/memory/sliding_window_memory.py)|

## 4. 使用方法
我们分别阐述不同的memory的用法。

### 4.1 WholeMemory
WholeMemory是全量记忆，存储所有消息。我们首先实例化了WholeMemory，然后通过`memory.add_message(msg)`方法将消息添加到memory中，并通过`memory.get_messages()`方法获取所有消息。

```python
from erniebot_agent.memory import AIMessage, HumanMessage, WholeMemory

memory = WholeMemory()
humanmessage = HumanMessage('请将这个图片中的单词识别出来')
aimessage = AIMessage('好，这个图片中的单词为meticulous') # Fake AIMessage
memory.add_message(humanmessage)
memory.add_message(aimessage)
humanmessage = HumanMessage('这个单词meticulous是什么意思呢？')
aimessage = AIMessage('这个单词的意思是挑剔的，关注细节的') # Fake AIMessage
memory.add_message(humanmessage)
memory.add_message(aimessage)
print(memory.get_messages())

>>> [<HumanMessage role: 'user', content: '请将这个图片中的单词识别出来', token_count: 14>, <AIMessage role: 'assistant', content: '好，这个图片中的单词为meticulous', token_count: 11>, <HumanMessage role: 'user', content: '这个单词meticulous是什么意思呢？', token_count: 21>, <AIMessage role: 'assistant', content: '这个单词的意思是挑剔的，关注细节的', token_count: 16>]
```
除了使用`memory.add_message(msg)`方法添加消息外，我们也可以通过`memory.add_messages(msgs)`方法批量添加消息。

```python
messages = [
    HumanMessage('请帮我把这个单词meticulous存储到单词本中'),
    AIMessage('好的，单词meticulous已经存储到单词本中'),
    HumanMessage('请问现在我的单词本中都有什么单词呢？'),
    AIMessage('单词中目前有单词：meticulous'),
]

memory.add_messages(messages)
print(memory.get_messages())

>>> [<HumanMessage role: 'user', content: '请将这个图片中的单词识别出来', token_count: 14>, <AIMessage role: 'assistant', content: '好，这个图片中的单词为meticulous', token_count: 11>, <HumanMessage role: 'user', content: '这个单词meticulous是什么意思呢？', token_count: 21>, <AIMessage role: 'assistant', content: '这个单词的意思是挑剔的，关注细节的', token_count: 16>, <HumanMessage role: 'user', content: '请帮我把这个单词meticulous存储到单词本中', token_count: 25>, <AIMessage role: 'assistant', content: '好的，单词meticulous已经存储到单词本中', token_count: 14>, <HumanMessage role: 'user', content: '请问现在我的单词本中都有什么单词呢？', token_count: 18>, <AIMessage role: 'assistant', content: '单词中目前有单词：meticulous', token_count: 9>]
```

如果在对话结束想要删除所有的消息记录，我们可以使用`memory.clear_chat_history()`方法。

```python
memory.clear_chat_history()
print(memory.get_messages())
```

### 4.2 LimitTokensMemory   
`LimitTokensMemory(max_token_limit)`是token数量限制截断记忆，在memory中存储固定数量的token。其中`max_token_limit`表示memory中最多存储的消息的token数量，超过这个限制后，从头开始对消息进行删除。

```python
from erniebot_agent.memory import  LimitTokensMemory
memory = LimitTokensMemory(max_token_limit=60)
memory.add_messages(messages)
print("裁剪前的消息为：", messages)
print("裁剪后的消息为：", memory.get_messages())

>>> 裁剪前的消息为： [<HumanMessage role: 'user', content: '请帮我把这个单词meticulous存储到单词本中', token_count: 25>, <AIMessage role: 'assistant', content: '好的，单词meticulous已经存储到单词本中', token_count: 14>, <HumanMessage role: 'user', content: '请问现在我的单词本中都有什么单词呢？', token_count: 18>, <AIMessage role: 'assistant', content: '单词中目前有单词：meticulous', token_count: 9>, <HumanMessage role: 'user', content: '我想对单词本中的单词全部打印出对应的中文含义用于记忆', token_count: 26>, <AIMessage role: 'assistant', content: '好的，单词本中包括：meticulous的意思是挑剔的，关注细节的', token_count: 21>]
裁剪后的消息为： [<AIMessage role: 'assistant', content: '单词中目前有单词：meticulous', token_count: 9>, <HumanMessage role: 'user', content: '我想对单词本中的单词全部打印出对应的中文含义用于记忆', token_count: 26>, <AIMessage role: 'assistant', content: '好的，单词本中包括：meticulous的意思是挑剔的，关注细节的', token_count: 21>]
```

### 4.3 SlidingWindowMemory   
`SlidingWindowMemory(max_round, retained_round)`是滑动窗口截断记忆，在memory中存储固定轮数的消息。其中`max_round`表示memory中消息最多存储的轮数，`retained_round`表示在memory中会保留的初始消息的轮数，用于初始消息比较重要的场景。下面两个例子我们分别展示SlidingWindowMemory保留一轮消息，和同时保留首轮消息的效果。

```python
from erniebot_agent.memory import SlidingWindowMemory

messages = [
    HumanMessage('请帮我把这个单词meticulous存储到单词本中'),
    AIMessage('好的，单词meticulous已经存储到单词本中'),
    HumanMessage('请问现在我的单词本中都有什么单词呢？'),
    AIMessage('单词中目前有单词：meticulous'),
    HumanMessage('我想对单词本中的单词全部打印出对应的中文含义用于记忆'),
    AIMessage('好的，单词本中包括：meticulous的意思是挑剔的，关注细节的'),
]

memory = SlidingWindowMemory(max_round=2, retained_round=0)
memory.add_messages(messages)
print("裁剪前的消息为：", messages)
print("裁剪后的消息为：", memory.get_messages())

>>> 裁剪前的消息为： [<HumanMessage role: 'user', content: '请帮我把这个单词meticulous存储到单词本中', token_count: 25>, <AIMessage role: 'assistant', content: '好的，单词meticulous已经存储到单词本中', token_count: 14>, <HumanMessage role: 'user', content: '请问现在我的单词本中都有什么单词呢？', token_count: 18>, <AIMessage role: 'assistant', content: '单词中目前有单词：meticulous', token_count: 9>, <HumanMessage role: 'user', content: '我想对单词本中的单词全部打印出对应的中文含义用于记忆', token_count: 26>, <AIMessage role: 'assistant', content: '好的，单词本中包括：meticulous的意思是挑剔的，关注细节的', token_count: 21>]
裁剪后的消息为： [<HumanMessage role: 'user', content: '请问现在我的单词本中都有什么单词呢？', token_count: 18>, <AIMessage role: 'assistant', content: '单词中目前有单词：meticulous', token_count: 9>, <HumanMessage role: 'user', content: '我想对单词本中的单词全部打印出对应的中文含义用于记忆', token_count: 26>, <AIMessage role: 'assistant', content: '好的，单词本中包括：meticulous的意思是挑剔的，关注细节的', token_count: 21>]
```

```python
from erniebot_agent.memory import SlidingWindowMemory

messages = [
    HumanMessage('请帮我把这个单词meticulous存储到单词本中'),
    AIMessage('好的，单词meticulous已经存储到单词本中'),
    HumanMessage('请问现在我的单词本中都有什么单词呢？'),
    AIMessage('单词中目前有单词：meticulous'),
    HumanMessage('我想对单词本中的单词全部打印出对应的中文含义用于记忆'),
    AIMessage('好的，单词本中包括：meticulous的意思是挑剔的，关注细节的'),
]

memory = SlidingWindowMemory(max_round=2, retained_round=1)
memory.add_messages(messages)
print("裁剪前的消息为：", messages)
print("裁剪后的消息为：", memory.get_messages())

>>>裁剪前的消息为： [<HumanMessage role: 'user', content: '请帮我把这个单词meticulous存储到单词本中', token_count: 25>, <AIMessage role: 'assistant', content: '好的，单词meticulous已经存储到单词本中', token_count: 14>, <HumanMessage role: 'user', content: '请问现在我的单词本中都有什么单词呢？', token_count: 18>, <AIMessage role: 'assistant', content: '单词中目前有单词：meticulous', token_count: 9>, <HumanMessage role: 'user', content: '我想对单词本中的单词全部打印出对应的中文含义用于记忆', token_count: 26>, <AIMessage role: 'assistant', content: '好的，单词本中包括：meticulous的意思是挑剔的，关注细节的', token_count: 21>]
裁剪后的消息为： [<HumanMessage role: 'user', content: '请帮我把这个单词meticulous存储到单词本中', token_count: 25>, <AIMessage role: 'assistant', content: '好的，单词meticulous已经存储到单词本中', token_count: 14>, <HumanMessage role: 'user', content: '我想对单词本中的单词全部打印出对应的中文含义用于记忆', token_count: 26>, <AIMessage role: 'assistant', content: '好的，单词本中包括：meticulous的意思是挑剔的，关注细节的', token_count: 21>]
```