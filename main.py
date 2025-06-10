import sys
from pathlib import Path
from typing import Dict

from core.ToolInterface import BaseTool
from rag import EmbeddingRAG
sys.path.append(str(Path(__file__).parent.parent))


from agents.AIAgent import AIAgent


# 定义一个自定义工具
class SalesDataTool(BaseTool):
    name: str = "sales_data"
    description: str = "获取矿泉水销售数据"
    
    def execute(self, params: Dict) -> str:
        # 这里实现获取销售数据的逻辑
        return "2023年矿泉水销售数据: 100万箱，在当前市场竞争激烈的环境下，矿泉水企业通常会通过提高产品质量、创新包装设计、增加产品线多样性（比如推出无糖、低钠等健康选择）、加强品牌建设和线上销售渠道拓展等方式来吸引消费者，提升市场份额。"
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
    description: str = "使用嵌入式RAG技术进行信息检索和生成"

    def execute(self, params: Dict) -> str:
        # 这里实现嵌入式RAG的逻辑
        RAGChat = EmbeddingRAG().chat()
        return "嵌入式RAG技术可以通过向量化文本数据来提高信息检索的效率和准确性。"
    pass
if __name__ == "__main__":


    # 初始化AIAgent对象
    agent = AIAgent("矿泉水销售专家", {})
    # 注册工具
    agent.register_tool(SalesDataTool())
    agent.register_tool(GraphTextTool())
    
    
    

    while True:
        query = input("用户输入: ")
        # 调用工具
        tool_result = agent.call_tool(["sales_data","book_graph_text"], query)
        response = agent.generate_response(
            agent.process_input(query,tool_result=tool_result)
        )
        print(f"助手回复: {response}")
