
demo文件结构如下所示：

  ```
  yiyan_plugin_demo/           # 插件demo注册的根目录
  |—.well-known
    |— ai-plugin.json         #插件主描述文件
    |— openapi.yaml          #插件API服务的标准描述文件
  |— logo.png               #插件的图标文件
  |— demo_server.py          #插件注册服务，可以启动到本地
  |— requirements.txt         #启动插件注册服务所依赖的库，要求python >= 3.7
  |— readme.md              # 说明文件
  ```


  * 第一步：修改ai-plugin.json文件：
  ```
  "name_for_human": "单词本", ===>   不能重名，可改成"单词本_zhangsan"或者"单词本_1024"（平台内全局唯一标识）
  "name_for_model": "wordbook", ===>   不能重名，可改成"wordbook_zhangsan"或者"单词本_1024"（平台内全局唯一标识）
  ```
  * 第二步：启动插件注册服务

  ```
  pip install -r requirement.txt
  python demo_server.py
  ```


  * 第三步：上传插件配置文件
    ![开发.png](https://bce.bdstatic.com/doc/eb118-guidbook/EB118-developer/%E5%BC%80%E5%8F%91_2519f45.png)