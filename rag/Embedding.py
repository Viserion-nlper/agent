import os
from typing import List
import numpy as np

import os
from dotenv import dotenv_values
env_variables = dotenv_values('.env')
for var in env_variables:
    print(var)
    os.environ[var] = env_variables[var]

class BaseEmbeddings:
    """
    Base class for embeddings
    """
    def __init__(self, path: str, is_api: bool) -> None:
        self.path = path
        self.is_api = is_api
    
    def get_embedding(self, text: str, model: str) -> List[float]:
        raise NotImplementedError

    def get_embeddings(self, text: List[str], model: str) -> List[List[float]]:
        raise NotImplementedError
    
    @classmethod
    def cosine_similarity(cls, vector1: List[float], vector2: List[float]) -> float:
        """
        calculate cosine similarity between two vectors
        """
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude
    
class ZhipuEmbedding(BaseEmbeddings):
    """
    class for Zhipu embeddings
    """
    def __init__(self, path: str = '', is_api: bool = True, embedding_dim = 2048) -> None:
        super().__init__(path, is_api)
        if self.is_api:
            from zhipuai import ZhipuAI
            self.client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY")) 
        self.embedding_dim = embedding_dim


    def get_embedding(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
        model="embedding-3",
        input=text,
        )
        return response.data[0].embedding