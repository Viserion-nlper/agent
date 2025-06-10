from typing import Dict, List

class MemoryModule:
    """记忆模块"""
    def __init__(self):
        self.history: List[Dict] = []

    def add_interaction(self, interaction: Dict[str, str]):
        # TUDO: 后面计划添加 20250610 21:33
        self.history.append(interaction)

    def get_recent(self, count: int = 5) -> List[Dict]:
       # TUDO: 后面计划添加 20250610 21:33
        return self.history[-count:]
