# File 模块介绍

## 1. File 简介

在建立一个Agent应用的过程中，由于LLM本身是无状态的，因此很重要的一点就是赋予Agent记忆能力。Agent的记忆能力主要可以分为长期记忆和短期记忆。

* 长期记忆通过文件/数据库的形式存储，是不会被遗忘的内容，每次判断需要相关知识就可以retrieval的方式，找到最相关的内容放入消息帮助LLM分析得到结果。
* 短期记忆则是直接在消息中体现，是LLM接触的一手信息，但是也受限于上下文窗口，更容易被遗忘。

这里我们简述在我们记录短期记忆的方式，即在Memory模块中存储消息，并通过消息裁剪，控制消息总条数不会超过上下文窗口。

在使用层面，Memory将传入Agent类中，用于记录多轮的消息，即Agent会在每一轮对话中和Memory有一次交互：即在LLM产生最终结论之后，将第一条HumanMessage和最后一条AIMessage加入到Memory中。

## 2. File 基类介绍

`File` 类是文件管理模块的基础类，用于表示通用的文件对象。它包含文件的基本属性，如文件ID、文件名、文件大小、创建时间、文件用途和文件元数据。此外，`File` 类还定义了一系列抽象方法，比较常用的有：异步读取文件内容的 `read_contents` 方法，以及将文件内容写入本地路径的 `write_contents_to` 方法以及一些辅助方法：生成文件的字符串表示形式、转换为字典形式等。在File类的内部，其主要有两个继承子类，一个是 `Local File`，一个是 `Remote File`。以下是 `File` 基类的属性以及方法介绍。

| 属性       | 类型           | 描述                                                      |
| ---------- | -------------- | --------------------------------------------------------- |
| id         | str            | 文件的唯一标识符                                          |
| filename   | str            | 文件名                                                    |
| byte_size  | int            | 文件大小（以字节为单位）                                  |
| created_at | str            | 文件创建时间的时间戳                                      |
| purpose    | str            | 文件的目的或用途，有"assistants", "assistants_output"两种 |
| metadata   | Dict[str, Any] | 与文件相关的附加元数据                                    |

| 方法              | 描述                           |
| ----------------- | ------------------------------ |
| read_contents     | 异步读取文件内容               |
| write_contents_to | 异步将文件内容写入本地路径     |
| get_file_repr     | 返回用于特定上下文的字符串表示 |
| to_dict           | 将File对象转换为字典           |

### 2.2 File 子类

#### 2.2.1 LocalFile 类

`LocalFile` 是 `File` 的子类，表示本地文件。除了继承自基类的属性外，它还添加了文件路径属性 `path`，用于表示文件在本地文件系统中的路径。

#### 2.2.2 RemoteFile 类

`RemoteFile` 也是 `File` 的子类，表示远程文件。它与 `LocalFile` 不同之处在于，它的文件内容存储在远程文件服务器交。`RemoteFile` 类还包含与远程文件服务器交互的相关逻辑。它还添加了文件服务器属性 `client`，用于表示文件的服务器。


## 3. FileManager 类介绍

`FileManager` 类是一个高级文件管理工具，封装了文件的创建、上传、删除等高级操作。它可能依赖于 `FileRegistry` 来管理文件的一致性和唯一性。`FileManager` 还包含了与远程文件服务器交互的逻辑，通过 `RemoteFileClient` 完成上传、下载、删除等文件操作。

## 4. RemoteFileClient 类介绍

`RemoteFileClient` 是用于与远程文件服务器交互的类。它定义了文件上传、文件下载、文件删除等操作的方法。`AIStudioFileClient` 是 `RemoteFileClient` 的一个具体实现，用于与 AI Studio 文件服务交互。

## 5. FileRegistry 类介绍

`FileRegistry` 是一个单例类，用于在整个应用程序中管理文件的注册和查找。它提供了方法来注册、注销、查找和列举已注册的文件。`FileRegistry` 通过全局唯一的实例 `_file_registry` 来跟踪应用程序中的所有文件。

## 6. 使用方法

1. 创建 `File` 对象，可以选择使用 `LocalFile` 或 `RemoteFile` 的子类。
2. 使用 `FileRegistry` 注册文件，确保文件的唯一性。
3. 使用 `FileManager` 进行高级文件操作，如上传、下载、删除等。

通过这一系列的类，文件管理模块提供了灵活、高效的文件管理解决方案，适用于本地文件和远程文件的处理。
