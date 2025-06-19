

import sys
import os
# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取项目根目录的路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
# 将项目根目录添加到模块搜索路径中
sys.path.insert(0, project_root)

from rag.knext.sub_command.ProjectCreator import projectCreator

# 创建ProjectCreator实例并调用create_project方法
project_creator_instance = projectCreator()

project_creator_instance.create_project(config_path="/Users/mac/Downloads/2025/客户/青源峰达/agent/rag/kag/examples/example_config.yaml")

