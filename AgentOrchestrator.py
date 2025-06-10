from typing import Dict
from BaseAgent import BaseAgent

class AgentOrchestrator:
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        
    def register_agent(self, agent: BaseAgent):
        """注册智能体实例"""
        self.agents[agent.profile.name] = agent
        
    def route_request(self, agent_name: str, request: str) -> Dict:
        """路由请求到指定智能体"""
        if agent_name not in self.agents:
            raise ValueError(f"Agent {agent_name} not registered")
        
        return self.agents[agent_name].process_request(request)