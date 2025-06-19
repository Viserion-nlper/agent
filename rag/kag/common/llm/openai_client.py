
from openai import OpenAI
import logging
from typing import Dict, Optional

logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

class OpenAIClient:
    """独立封装的OpenAI客户端和流式响应"""
    
    @classmethod
    def from_config(cls, config: Dict) -> 'OpenAIClient':
        """
        从配置字典中创建并返回一个OpenAIClient实例。

        Args:
            config (Dict): 包含配置信息的字典。

        Returns:
            OpenAIClient: 初始化后的OpenAIClient实例。
        """
        api_key = config.get('api_key')
        base_url = config.get('base_url')
        model = config.get('model')
        
        #TODO type用于区分不同的服务类型，后面可能要添加多模态的client  这里预留0619
        service_type = config.get('type')
        
        client_instance = cls(
            api_key=api_key,
            base_url=base_url,
            model=model,
            stream=False,  # 假设默认不使用流式响应，可以根据需要调整
            temperature=0.7,  # 假设默认温度为0.7，可以根据需要调整
            timeout=None  # 假设默认没有超时设置，可以根据需要调整
        )
        
        return client_instance
    
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        stream: bool = False,
        temperature: float = 0.7,
        timeout: Optional[float] = None,
        is_azure: bool = False
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.stream = stream
        self.temperature = temperature
        self.timeout = timeout
        self.client = (
            OpenAI(api_key=api_key, base_url=base_url)
        )

    def __call__(self, prompt: str, image_url: Optional[str] = None) -> str:
        messages = [{
            "role": "system", 
            "content": "you are a helpful assistant"
        }]
        
        if image_url:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            })
        else:
            messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=self.stream,
            temperature=self.temperature,
            timeout=self.timeout
        )

        if not self.stream:
            return response.choices[0].message.content
        
        return "".join(
            chunk.choices[0].delta.content 
            for chunk in response 
            if chunk.choices[0].delta.content
        )

