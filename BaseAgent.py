
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from enum import Enum
import requests
from pydantic import BaseModel, Field
import json

"""没有重构前的代码，这里用作保存，以后再删除"""
class BaseTool(BaseModel):
    """工具基类"""
    name: str
    description: str
    
    @abstractmethod
    def execute(self, params: Dict) -> str:
        pass
class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "claude"
    LOCAL = "local"

class APIPermission(BaseModel):
    endpoint: str
    methods: List[str]
    rate_limit: int

class AgentProfile(BaseModel):
    name: str
    version: str
    description: str
    system_prompt: str = Field(default_factory=str)  # 使用alias来处理可能的拼写错误
    metadata: Dict[str, str] = Field(default_factory=dict)  # 添加默认工厂函数以处理可能为空的情况
    capabilities: List[str] = Field(default_factory=list)  # 添加默认工厂函数
    supported_languages: List[str] = Field(default_factory=list)  # 添加默认工厂函数
    constants: str = Field(default_factory=str)  # 添加默认工厂函数
    dependencies: Dict[str, str] = Field(default_factory=dict)  # 添加默认工厂函数
    example: List[Dict[str, str]] = Field(default_factory=list)  # 添加默认工厂函数
    

class BaseAgent(ABC):
    """Agent抽象基类"""
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.memory = []
        self._initialize_components()


    
    @abstractmethod
    def _initialize_components(self):
        """profile定义"""
        self.profile = {
            'name':'default_agent',
            'version':'1.0',
            'system_propmt':'你是一个擅长于矿泉水销售及市场营销的专家, 请根据下文中给出的example进行学习分析，并且根据用户提问的问题进行发散性回答',
            'description':'这是一个默认的Agent实现',
            'capabilities': ['text_processing', 'response_generation', 'tool_calling'],
            'supported_languages': ['zh', 'en'],
            'constants': {
                '约束如下：要生成一些可支撑的数据来验证你的观点。'
            },
            'dependencies': {
                'nlu': 'LanguageUnderstandingModule',
                'memory': 'MemoryModule',
                'response_generator': 'ResponseGenerator',
                'tool_agent': 'ToolCallAgent'
            },
            'example': [{
                "user": "你好",
                "bot": "你好，有什么可以帮到你的？"
            }]
        }
        """调用的API列表"""
        self.apiList: dict[str] = {}
        """LLM列表"""
        self.llmList: List[Dict] = []

        """记忆模块"""
        # self.memory =VectorMemory()

        """权限控制配置"""
        self.permissions = {
            'roles': ['user', 'admin'],
            'permissions': {
                'admin': ['full_access'],
                'user': ['basic_query']
            }
        }

        """初始化核心组件"""
        self.tool_agent = ToolCallAgent()

    @abstractmethod
    def process_input(self, input_text: str) -> Dict[str, Any]:
        """处理用户输入"""
        pass

    @abstractmethod
    def generate_response(self, processed_data: Dict[str, Any]) -> str:
        """生成响应"""
        pass
        
    
    def register_tool(self, tool: BaseTool):
        """注册工具到智能体"""
        self.tool_agent.register_tool(tool)
        
    def call_tool(self, tools_name: List[str], parameters: Dict) -> str:
        """判断需要加载的工具是否都被注册"""
        for tool in tools_name:
            # 如果tool_name中的某个tool不在self.tool_agent.available_tools中，则执行相应操作
            if tool not in list(self.tool_agent.available_tools.keys()):
                # 如果tool_name中的某个tool不在self.tool_agent.available_tools中，则执行相应操作（此处省略具体操作）
                raise ValueError(f"Tool {tool} is not registered")
            
        
        return self.tool_agent.run(f"{json.dumps(parameters,ensure_ascii=False)}")

class LanguageUnderstandingModule:
    """语言理解模块"""
    def analyze(self, text: str) -> Dict[str, Any]:
        """分析文本，返回意图和实体信息"""
        text = text.strip()
        return text

class MemoryModule:
    """记忆模块"""
    def __init__(self):
        self.history = []

    def add_interaction(self, interaction: Dict[str, str]):
        self.history.append(interaction)

class ResponseGenerator:
    """响应生成模块"""
    def generate(self, context: Dict[str, Any]) -> str:
        response = requests.post('http://localhost:11434/api/generate', json=context)
        # print(json.loads(response.text).get("response"))
        return json.loads(response.text).get("response")


class ToolCall(BaseModel):
    """工具调用元数据"""
    name: List[str] = Field(..., description="工具名称，可以是单个工具或多个工具的组合")
    parameters: Dict
    call_id: str



class ToolCallAgent:
    def __init__(self):
        self.available_tools: Dict[str, BaseTool] = {}
        self.memory: List[Dict] = []
        self.max_react_cycles = 5
        self.response_generator = ResponseGenerator()
        
    def _make_llm_decision(self, query: str) -> Optional[Dict]:
        """使用LLM决定是否调用工具"""
        tools_prompt = "\n".join(
            f"- 工具名称是：{tool.name}: 该工具的介绍是：{tool.description}" 
            for tool in self.available_tools.values()
        )
        
        llm_prompt = f"""根据用户查询和可用工具，决定是否需要调用工具。
            用户查询: {query}

            可用工具:
            {tools_prompt}

            请严格按以下JSON格式响应：
            {{
                "need_tool": bool,
                "tool_name": List[str]或者 str,
                "parameters": dict
            }}
            tool_name有可能是单个tool也有可能是多个tool的组合，
            例如：
            {{
                "need_tool": true,
                "tool_name": "weather",
                "parameters": {{"location": "Beijing"}}
            }}
            或者
            {{
                "need_tool": true,
                "tool_name": ["weather", "database"],
                "parameters": {{"location": "Beijing", "query": "sales"}}
            }}"""

        response = self.response_generator.generate({
            "model": "qwen2",
            "prompt": llm_prompt,
            "temperature": 0.7,
            "stream": False
        })
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return None

    def register_tool(self, tool: BaseTool):
        """注册工具到智能体"""
        self.available_tools[tool.name] = tool

    def _think(self, query: str) -> Optional[ToolCall]:
        """ReAct思考阶段：使用LLM决定是否调用工具"""
        decision = self._make_llm_decision(query)
        if decision and decision.get("need_tool", False):
            return ToolCall(
                name=decision["tool_name"],
                parameters=decision["parameters"],
                call_id=f"call_{len(self.memory)}"
            )
        return None

    def _act(self, tool_call: ToolCall) -> str:
        """ReAct执行阶段：运行工具并返回结果"""

        sumary_tool_result = ""
        if isinstance(tool_call.name, str):
            tool = self.available_tools.get(tool_call.name)
            if not tool:
                return f"Error: Tool {tool_call.name} not found"
            try:
                tool_result = tool.execute(tool_call.parameters)
                self.memory.append({
                    "call_id": tool_call.call_id,
                    "result": tool_result
                })
                return f"Tool {tool.name} executed with result: {tool_result}"
            except Exception as e:
                    return f"Tool execution failed: {str(e)}"
        elif isinstance(tool_call.name, list):
            for t in tool_call.name:
                tool = self.available_tools.get(t)
                if not tool:
                    return f"Error: Tool {tool_call.name} not found"
                try:
                    tool_result = tool.execute(tool_call.parameters)
                    self.memory.append({
                        "call_id": tool_call.call_id,
                        "result": tool_result
                    })
                    sumary_tool_result += f"Tool {tool.name} executed with result: {tool_result}\n"
                except Exception as e:
                    return f"Tool execution failed: {str(e)}"
            return sumary_tool_result
        
        
       

    def run(self, query: str) -> str:
        summary_result = ""
        """执行ReAct循环"""
        for _ in range(self.max_react_cycles):
            tool_call = self._think(query)
            if not tool_call:
                break # 如果不需要调用工具，则结束循环
            
            result = self._act(tool_call)
            # 此处可添加结果处理逻辑
            query = result  # 将结果作为下一轮输入
            summary_result += f"Tool {tool_call.name} executed with result: {result}\n" 
            # TD: 这里可以添加对结果的进一步处理，比如将结果存入memory模块，或者使用llm来总结生成最终响应
            with open("tool_call_log.txt", "a", encoding="utf-8") as f:
                f.write(f"Tool {tool_call.name} executed with result: {result}\n")
        return summary_result
class AIAgent(BaseAgent):
    """具体Agent实现"""
    def _initialize_components(self):
        self.nlu = LanguageUnderstandingModule()
        self.memory = MemoryModule()
        self.generator = ResponseGenerator()
        self.tool_agent = ToolCallAgent()
        self.profile = AgentProfile(
            name="default_agent",
            version="1.0",
            description="这是一个默认的Agent实现",
   
            system_prompt=f"你是一个擅长于{self.name}, 请根据下文中给出的example进行学习分析，并且根据用户提问的问题进行发散性回答",
            metadata={"author": "default"},
            constants="约束如下：不要在回答中使用任何不恰当或冒犯人的词汇。",
            dependencies={
                "nlu": "LanguageUnderstandingModule",
                "memory": "MemoryModule",
                "response_generator": "ResponseGenerator",
                "tool_agent": "ToolCallAgent"
            },
            example=[{
                "user": "你好",
                "bot": "你好，有什么可以帮到你的？"
            }],
            capabilities=["text_processing", "response_generation", "tool_calling"],
            supported_languages=["zh"],
        ),
        self.apiList = {
            "weather": APIPermission(
                endpoint="https://api.weather.com/v1",
                methods=["GET"],
                rate_limit=100
            ),
            "database": APIPermission(
                endpoint="https://internal.db.api",
                methods=["GET", "POST"],
                rate_limit=50
            )
        }
        self.llmList = [
            {
                "name": "qwen2",
                "provider": LLMProvider.LOCAL,
                "config": {"temperature": 0.7}
            },
            {
                "name": "GPT-4-turbo",
                "provider": LLMProvider.OPENAI,
                "config": {"api_key": "sk-***"}
            }
        ]
        self.permissions = {
            'roles': ['user', 'admin'],
            'permissions': {
                'admin': ['full_access'],
                'user': ['basic_query']
            }
        }


    def process_input(self, input_text: str, tool_result: str) -> Dict[str, Any]:
        """拿到agent定义的LLM列表，进行处理"""

        avilable_llmList = self.llmList[0]
        avilable_system_prompt = self.profile[0].system_prompt
        avilable_constants = self.profile[0].constants
        avilable_example = self.profile[0].example
        avilable_capabilities = self.profile[0].capabilities
        avilable_supported_languages = self.profile[0].supported_languages


        # 
        # TODO: 这里可以实现query改写，或者进行历史问答对话相似性检索，这里先不做处理，按照原输入文本进行返回
        analysised_inpute_text = self.nlu.analyze(input_text)

        analysis = {
                "model": avilable_llmList.get("name"),
                "prompt": f'''
                    {avilable_system_prompt}\n\n
                    用户的问题问题是：{analysised_inpute_text}\n\n
                    约束如下：{avilable_constants}\n\n
                    请参考给出的示例来进行分析：{avilable_example}\n\n
                    其中调用工具进行处理的结果是：{tool_result}\n\n
                    请特别参考调用工具处理的结果进行分析\n\n
                    请根据用户的问题进行发散性回答，并且使用{avilable_supported_languages}语言来进行回答。       
                    ''',
                "stream": False,
                "temperature": avilable_llmList.get("config").get("temperature", 0.7),
            }
        


        # self.memory.add_interaction({
        #     "input": input_text,
        #     "analysis": analysis
        # })

        return analysis

    def generate_response(self, processed_data: Dict[str, Any]) -> str:
        # TODO 这里可以对处理后的数据进行进一步处理，比如添加一些上下文信息
        return self.generator.generate(processed_data)
