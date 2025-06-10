import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from agents.AIAgent import AIAgent

if __name__ == "__main__":
    agent = AIAgent("销售助手", {})
    while True:
        query = input("用户输入: ")
        response = agent.generate_response(
            agent.process_input(query)
        )
        print(f"助手回复: {response}")
