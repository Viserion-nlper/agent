from abc import ABC, abstractmethod
from typing import Any, Dict, List

class BaseAgent(ABC):
    """Agent核心抽象类"""
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.memory = []
        self._initialize_components()

    @abstractmethod
    def _initialize_components(self):
        """初始化组件"""
        pass

    @abstractmethod 
    def process_input(self, input_text: str) -> Dict[str, Any]:
        """处理用户输入"""
        pass

    @abstractmethod
    def generate_response(self, processed_data: Dict[str, Any]) -> str:
        """生成响应"""
        pass

    @abstractmethod
    def register_tool(self, tool):
        """注册工具"""
        pass

    @abstractmethod
    def call_tool(self, tool_name: str, params: Dict) -> str:
        """调用工具"""
        pass
