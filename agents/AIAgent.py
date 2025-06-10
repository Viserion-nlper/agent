from core.AgentCore import BaseAgent
from components.LanguageProcessor.LanguageModule import LanguageUnderstandingModule
from components.MemorySystem.MemoryModule import MemoryModule
from components.ResponseGenerator.ResponseModule import ResponseGenerator
from components.ToolManager.ToolAgent import ToolCallAgent
from typing import Dict, Any, List
import json
from enum import Enum

class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "claude"
    LOCAL = "local"

class AIAgent(BaseAgent):
    """Agent实现类"""
    def _initialize_components(self):
        """初始化所有组件"""
        self.nlu = LanguageUnderstandingModule()
        self.memory = MemoryModule()
        self.generator = ResponseGenerator()
        self.tool_agent = ToolCallAgent()
        
        # 初始化配置
        self.profile = {
            'name': 'default_agent',
            'version': '1.0',
            'system_prompt': f'你是一个擅长于{self.name}的专家',
            'description': '这是一个默认的Agent实现',
            'capabilities': ['text_processing', 'response_generation', 'tool_calling'],
            'supported_languages': ['zh'],
            'constants': '约束如下：要生成一些可支撑的数据来验证你的观点。',
            'example': [{
                "user": "你好",
                "bot": "你好，有什么可以帮到你的？"
            }]
        }

    def process_input(self, input_text: str, tool_result: str = "") -> Dict[str, Any]:
        """处理用户输入"""
        analysis = {
            "model": "qwen2",
            "prompt": f'''
                {self.profile['system_prompt']}\n
                用户问题: {self.nlu.analyze(input_text)}\n
                约束: {self.profile['constants']}\n
                示例: {self.profile['example']}\n
                {f"工具结果: {tool_result}\n" if tool_result else ""}
                请用{self.profile['supported_languages']}回答
            ''',
            "temperature": 0.7,
            "stream": False
        }
        return analysis

    def generate_response(self, processed_data: Dict[str, Any]) -> str:
        """生成响应"""
        return self.generator.generate(processed_data)

    def register_tool(self, tool):
        """注册工具"""
        self.tool_agent.register_tool(tool)

    def call_tool(self, tool_name: str, params: Dict) -> str:
        """调用工具"""
        return self.tool_agent.run(
            f"调用工具{tool_name}，参数: {json.dumps(params, ensure_ascii=False)}"
        )
