
import unittest
from typing import Dict
from pydantic import ValidationError
from BaseAgent import BaseAgent, ToolCallAgent, AgentProfile

class TestAgentFramework(unittest.TestCase):
    """框架核心组件验证测试"""
    
    def test_profile_validation(self):
        """测试Profile数据验证"""
        # 合法配置
        valid_profile = AgentProfile(
            name="test_agent",
            version="1.0",
            description="测试智能体",
            system_prompt="你是一个测试AI",
            metadata={"author": "tester"}
        )
        self.assertEqual(valid_profile.name, "test_agent")
        
        # 非法配置
        with self.assertRaises(ValidationError):
            AgentProfile(name="")  # 空名称应触发验证错误

    # def test_llm_switching(self):
    #     """测试多模型切换功能"""
    #     agent = ToolCallAgent()
    #     # 验证默认LLM加载
    #     self.assertGreater(len(agent.llm_list), 0)
        
    #     # 切换模型测试
    #     original_llm = agent.current_llm
    #     agent.switch_llm("ERNIE-4.0")
    #     self.assertNotEqual(original_llm, agent.current_llm)

    # def test_api_permission(self):
    #     """测试API权限控制"""
    #     agent = ToolCallAgent()
    #     # 白名单验证
    #     self.assertTrue(agent.check_api_permission("weather"))
    #     # 黑名单验证
    #     self.assertFalse(agent.check_api_permission("admin_api"))

    def test_end_to_end_processing(self):
        """端到端流程测试"""
        agent = ToolCallAgent()
        test_input = "查询北京天气"
        response = agent.process_request(test_input)
        
        # 验证响应结构
        self.assertIn("status", response)
        self.assertIn("content", response)
        self.assertEqual(response["status"], "success")

class IntegrationTest(unittest.TestCase):
    """系统集成测试"""
    
    def test_orchestrator_routing(self):
        """测试协调器路由功能"""
        from your_module import AgentOrchestrator
        
        orchestrator = AgentOrchestrator()
        test_agent = ToolCallAgent()
        orchestrator.register_agent(test_agent)
        
        response = orchestrator.route_request(
            agent_name=test_agent.profile.name,
            request="需要帮助"
        )
        self.assertIsInstance(response, Dict)

if __name__ == "__main__":
    unittest.main()
