# Agent 系统架构文档

## 模块结构
```
agent/
├── core/               # 核心抽象定义
│   ├── AgentCore.py    # Agent抽象基类
│   └── ToolInterface.py # 工具接口定义
├── components/         # 功能组件
│   ├── LanguageProcessor/
│   │   └── LanguageModule.py
│   ├── MemorySystem/
│   │   └── MemoryModule.py
│   ├── ResponseGenerator/
│   │   └── ResponseModule.py
│   └── ToolManager/
│       └── ToolAgent.py
├── agents/             # 具体Agent实现
│   └── AIAgent.py
└── main.py             # 程序入口
```

## 核心模块说明

### 1. AgentCore (BaseAgent)
- 定义Agent基本行为和接口
- 抽象方法包括：
  - `process_input()`: 处理用户输入
  - `generate_response()`: 生成响应
  - `register_tool()`: 注册工具
  - `call_tool()`: 调用工具

### 2. 功能组件
- **LanguageProcessor**: 自然语言理解
- **MemorySystem**: 对话记忆管理
- **ResponseGenerator**: 响应生成
- **ToolManager**: 工具调用管理

## 使用示例

```python
from agents.AIAgent import AIAgent
from core.ToolInterface import BaseTool

class WeatherTool(BaseTool):
    name = "weather"
    description = "查询天气"
    
    def execute(self, params):
        return f"{params['location']}天气晴朗"

agent = AIAgent("天气预报助手", {})
agent.register_tool(WeatherTool())

response = agent.generate_response(
    agent.process_input("北京天气怎么样？")
)
print(response)
```

## 重构总结
1. 将单体架构拆分为模块化组件
2. 保留原有全部业务逻辑
3. 各模块职责单一、接口清晰
4. 便于后续扩展和维护
