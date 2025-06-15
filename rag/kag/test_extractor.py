import json
import unittest

import requests
from schema_based_extractor import SchemaBasedExtractor
from example_schema import SCHEMA_DEFINITIONS

class MockLLMClient:
    """模拟LLM客户端用于测试"""
    
    def query(self, prompt):
        
        analysis = {
            "model": "qwen2",
            "prompt": prompt,
            "temperature": 0.7,
            "stream": False
        }
        response = requests.post(
                    'http://localhost:11434/api/generate',
                    json=analysis
                )
        return json.loads(response.text).get("response", "")

class TestSchemaBasedExtractor(unittest.TestCase):
    
    def setUp(self):
        self.llm = MockLLMClient()
        self.extractor = SchemaBasedExtractor(SCHEMA_DEFINITIONS, self.llm)
    
    def test_entity_recognition(self):
        text = "马云是阿里巴巴集团的创始人"
        entities = self.extractor.named_entity_recognition(text)
        self.assertEqual(len(entities), 2)
        self.assertEqual(entities[0]["name"], "马云")
    
    def test_entity_standardization(self):
        entities = [{"name": "马云", "type": "人物"}]
        standardized = self.extractor.entity_standardization("", entities)
        self.assertEqual(standardized[0]["id"], "人物_马云")
    
    def test_relation_extraction(self):
        entities = [
            {"id": "人物_马云", "name": "马云", "type": "人物"},
            {"id": "公司_阿里巴巴集团", "name": "阿里巴巴集团", "type": "公司"}
        ]
        relations = self.extractor.relation_extraction("", entities)
        self.assertTrue(any(r["predicate"] == "创始人" for r in relations))
    
    def test_full_pipeline(self):
        text = "马云是阿里巴巴集团的创始人"
        result = self.extractor.extract_triples(text)
        self.assertEqual(len(result["entities"]), 2)
        self.assertEqual(len(result["relations"]), 1)

if __name__ == "__main__":
    unittest.main()