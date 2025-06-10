import requests
import json
from typing import Dict, Any

class ResponseGenerator:
    """响应生成模块"""
    def generate(self, context: Dict[str, Any]) -> str:
        """生成响应"""
        response = requests.post(
            'http://localhost:11434/api/generate',
            json=context
        )
        return json.loads(response.text).get("response", "")
