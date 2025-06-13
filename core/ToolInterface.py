from pydantic import BaseModel, Field
from typing import Dict, List
from abc import ABC, abstractmethod

class BaseTool(BaseModel, ABC):
    """工具抽象基类"""
    name: str
    description: str
    
    @abstractmethod
    def execute(self, params: Dict) -> str:
        """执行工具"""
        pass

class ToolCall(BaseModel):
    """工具调用元数据"""
    name: List[str] = Field(..., description="工具名称，可以是单个工具或多个工具的组合")
    parameters: Dict
    query: str = Field(..., description="用户查询")
    call_id: str
