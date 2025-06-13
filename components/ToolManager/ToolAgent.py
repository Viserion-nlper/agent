from typing import Dict, List, Optional
import json
from ..ResponseGenerator.ResponseModule import ResponseGenerator
from core.ToolInterface import BaseTool, ToolCall

class ToolCallAgent:
    """工具调用管理模块"""
    def __init__(self):
        self.available_tools: Dict[str, BaseTool] = {}
        self.memory: List[Dict] = []
        self.max_react_cycles = 5
        self.response_generator = ResponseGenerator()

    def register_tool(self, tool: BaseTool):
        """注册工具"""
        self.available_tools[tool.name] = tool

    def _make_llm_decision(self, query: str) -> Optional[Dict]:
        """使用LLM决定是否调用工具"""
        # self.available_tools.values()是注册的所有工具的列表
        tools_prompt = "\n".join(
            f"- {tool.name}: {tool.description}" 
            for tool in self.available_tools.values()
        )
        
        llm_prompt = f"""
        根据用户查询和可用工具，决定是否需要调用工具。
        用户查询: {query} \n
        可用工具列表:
        {tools_prompt} \n

        请严格按以下JSON格式响应：
        {{
            "need_tool": bool,
            "tool_name": List[str]或者 str,
            "parameters": dict
        }}\n
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
        }}
        """
        
        
        response = self.response_generator.generate({
            "model": "qwen2",
            "prompt": llm_prompt,
            "temperature": 0.7,
            "stream": False
        })
        response = json.loads(response)
        if not isinstance(response, dict):
            return None
        # 确保返回的JSON格式正确
        else:
            response.update({"query":query})
        try:
            return response
        except json.JSONDecodeError:
            return None

    def _think(self, query: str) -> Optional[ToolCall]:
        """决定是否调用工具"""
        decision = self._make_llm_decision(query)
        if decision and decision.get("need_tool", False):
            return ToolCall(
                name=decision["tool_name"],
                parameters=decision["parameters"],
                query=decision.get("query", query),
                call_id=f"call_{len(self.memory)}"
            )
        return None

    def _act(self, tool_call: ToolCall) -> str:
        """执行工具调用"""
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
                return tool_result
            except Exception as e:
                return f"Tool execution failed: {str(e)}"
        elif isinstance(tool_call.name, list):
            results = []
            for tool_name in tool_call.name:
                tool = self.available_tools.get(tool_name)
                if not tool:
                    return f"Error: Tool {tool_name} not found"
                try:
                    tool_result = tool.execute(tool_call.query)
                    results.append(tool_result)
                except Exception as e:
                    return f"Tool {tool_name} execution failed: {str(e)}"
            return "\n".join(results)

    def run(self, query: str) -> str:
        summary_result = ""
        """执行工具调用流程"""
        for _ in range(self.max_react_cycles):
            tool_call = self._think(query)
            if not tool_call:
                break
            result = self._act(tool_call)
            query = result  # 将结果作为下一轮输入
            summary_result += f"Tool {tool_call.name} executed with result: {result}\n" 
            # TD: 这里可以添加对结果的进一步处理，比如将结果存入memory模块，或者使用llm来总结生成最终响应
            with open("tool_call_log.txt", "a", encoding="utf-8") as f:
                f.write(f"Tool {tool_call.name} executed with result: {result}\n")
        return summary_result
