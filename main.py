import sys
from pathlib import Path
from typing import Dict

from core.ToolInterface import BaseTool
from rag import EmbeddingRAG
from rag.qwen3_Embedding_milvus import query_with_rag
sys.path.append(str(Path(__file__).parent.parent))


from agents.AIAgent import AIAgent


# 定义一个自定义工具1
class SalesDataTool(BaseTool):
    name: str = "sales_data"
    description: str = "获取矿泉水销售数据"
    
    def execute(self, params: Dict) -> str:
        # 这里实现获取销售数据的逻辑
        return "2023年矿泉水销售数据: 100万箱。"
# 定义一个自定义工具2
class GraphTextTool(BaseTool):
    name: str = "book_graph_text"
    description: str = "获取无糖、低钠矿泉水销售数据"
    
    def execute(self, params: Dict) -> str:
        # 这里实现获取书籍图书馆数据
        return "2025年图书馆书籍内容：《Python编程从入门到实践》"

# 定义EmbeddingRAG类
class RAGTool(BaseTool):
    """嵌入式RAG工具"""
    name: str = "embedding_rag"
    description: str = "使用嵌入式RAG技术进行信息检索和生成，其中包含了项目的预算情况等相关信息"
    def execute(self, params: Dict) -> str:
        # 这里实现嵌入式RAG的逻辑
        from pymilvus import MilvusClient
        milvus_client = MilvusClient(uri="./milvus_demo.db")
        collection_name = "my_rag_collection"
        rag_result = query_with_rag(milvus_client, collection_name, params)
        return rag_result
if __name__ == "__main__":
    # 初始化AIAgent对象
    agent = AIAgent("矿泉水软件产品销售专家", {})
    # 注册工具
    agent.register_tool(SalesDataTool())
    agent.register_tool(GraphTextTool())
    agent.register_tool(RAGTool())
    
    while True:
        query = input("用户输入: ")
        # 调用工具
        tool_result = agent.call_tool(query)
        # 生成响应
        response = agent.generate_response(
            agent.process_input(query,tool_result=tool_result)
        )
        print(f"助手回复: {response}")
