import glob
import os


def recursively_search_yaml(directory):
    yaml_files = []
    # 递归搜索指定目录及其子目录下的所有 YAML 文件
    for root, dirs, files in os.walk(directory):
        for file in glob.glob(os.path.join(root, "*.yaml")):
            yaml_files.append(file)
            print(f"Found YAML file: {file}")
            # 读取 YAML 文件

    return yaml_files


def relative_import(module_name, submodule_name):
    import importlib

    # 动态导入模块
    module = importlib.import_module(module_name)

    # 获取模块中的子模块/对象
    submodule = getattr(module, submodule_name)

    return submodule


if __name__ == "__main__":
    # 使用示例
    recursively_search_yaml("./benchmark/tasks")
