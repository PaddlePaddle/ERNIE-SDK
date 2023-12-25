# File 模块介绍

## 1. File 简介

文件管理模块提供了用于管理文件的一系列类方便用户与Agent进行交互，其中包括 `File` 及其子类、`FileManager` 、`GlobalFileManagerHandler`以及与远程文件服务器交互的  `RemoteFileClient`。推荐使用 `GlobalFileManagerHandler`在事件循环开始时初始化 `FileManager`以及获取全局的 `FileManager`，之后只需通过这个全局的 `FileManager`对文件进行增、删、查等操作，**不推荐**用户自行操作 `File`类以免造成资源泄露，同时从效率方面考虑，`FileManager`将作为此模块中生命周期最长的对象，它会在关闭时回收所有的持有对象（RemoteClient/temp local file），请不要随意关闭它。

## 2. File 基类介绍

`File` 类是文件管理模块的基础类，用于表示通用的文件对象（不建议自行创建 `File` 类以免无法被 `Agent`识别）。它包含文件的基本属性，如文件ID、文件名、文件大小、创建时间、文件用途和文件元数据。此外， `File `类还定义了一系列抽象方法，比较常用的有：异步读取文件内容的 `read_contents `方法，以及将文件内容写入本地路径的 `write_contents_to `方法以及一些辅助方法：生成文件的字符串表示形式、转换为字典形式等。在File类的内部，其主要有两个继承子类，一个是 `Local File `，一个是 `Remote File `。以下是 `File` 基类的属性以及方法介绍。

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

`RemoteFile` 也是 `File` 的子类，表示远程文件。它与 `LocalFile` 不同之处在于，它的文件内容存储在远程文件服务器交。`RemoteFile` 类还包含与远程文件服务器交互的相关逻辑。

## 3. FileManager 类介绍

`FileManager` 类是一个高级文件管理工具，封装了文件的创建、上传、删除等高级操作，以及与 `Agent`进行交互，无论是  `Local File `还是 `RemoteFIle `都可以使用它来统一管理。`FileManager `集成了与远程文件服务器交互的逻辑，通过 `RemoteFileClient `完成上传、下载、删除等文件操作以及与本地文件交互的逻辑，从本地地址创建 `Local File `。它依赖于 `FileRegistry` 来对文件进行用于在整个应用程序中管理文件的注册和查找。

以下是相关的属性和方法

| 属性               | 类型               | 描述                     |
| ------------------ | ------------------ | ------------------------ |
| remote_file_client | RemoteFileClient   | 远程文件客户端。         |
| save_dir           | Optional[FilePath] | 用于保存本地文件的目录。 |
| closed             | bool               | 文件管理器是否已关闭。   |

| 方法                         | 描述                                 |
| ---------------------------- | ------------------------------------ |
| create_file_from_path        | 从指定文件路径创建文件               |
| create_local_file_from_path  | 从文件路径创建本地文件               |
| create_remote_file_from_path | 从文件路径创建远程文件并上传至服务器 |
| create_file_from_bytes       | 从字节创建文件                       |
| retrieve_remote_file_by_id   | 通过ID获取远程文件                   |
| look_up_file_by_id           | 通过ID查找本地文件                   |
| list_remote_files            | 列出远程文件                         |

注：

* `FileManager` 类不可被复制以免造成资源泄露。
* 如果未指定 `save_dir`，那么当 `FileManager`关闭时，所有与之关联的本地文件都会被回收。
* 如果 `FileManager` 类有相关联的 `RemoteFileClient`，那么当 `FileManager`关闭时，相关联的 `RemoteFileClient`也会一起关闭。

## 4. RemoteFileClient 类介绍

`RemoteFileClient` 是用于与远程文件服务器交互的类。它定义了文件上传、文件下载、文件删除等操作的方法。`AIStudioFileClient` 是 `RemoteFileClient` 的一个具体推荐实现，用于与文件服务交互，用户使用 `access token`作为参数用于身份验证，之后能够在AIStudio文件服务中上传、检索、列出文件，以及创建临时URL以访问文件。`RemoteFileClient`使用时被 `FileManager`持有，一旦 `FileManager`关闭，`RemoteFileClient`也会相应被关闭，其中的资源也会被相应释放。

## 5. 使用方法

1. 通过 `GlobalFileManagerHandler`获取全局的FileManager，通过它来控制所有文件，注：它的生命周期同整个事件循环。

```python
from erniebot_agent.file import GlobalFileManagerHandler
async def demo_function():
    file_manager = await GlobalFileManagerHandler().get()  
```

2. 通过 `GlobalFileManagerHandler`创建 `File`

```python
from erniebot_agent.file import GlobalFileManagerHandler
async def demo_function():
    file_manager = await GlobalFileManagerHandler().get()
    # 从路径创建File, file_type可选择local或者remote file_purpose='assistant'代表用于给LLM输入使用
    local_file = await file_manager.create_file_from_path(file_path='your_path', file_type='local')
```

3. 通过 `GlobalFileManagerHandler`搜索 `File`

```python
from erniebot_agent.file import GlobalFileManagerHandler
async def demo_function():
    file_manager = await GlobalFileManagerHandler().get()
    file = file_manager.look_up_file_by_id(file_id='your_file_id')
    file_content = await file.read_contents()
```
