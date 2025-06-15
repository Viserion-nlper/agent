import copy
import json
import logging
from typing import Dict, List

import requests


class SchemaBasedExtractor:
    """
    基于Schema的三元组抽取器
    实现从文本中抽取符合预定义Schema的实体和关系
    """
    
    def __init__(self, schema, llm_client):
        """
        初始化抽取器
        Args:
            schema: 预定义的Schema对象
            llm_client: LLM客户端用于文本处理
        """
        self.schema = schema
        self.llm_client = llm_client
        self.logger = logging.getLogger(__name__)
    
    def named_entity_recognition(self, text: str) -> List[Dict]:
        """
        命名实体识别
        Args:
            text: 输入文本
        Returns:
            List[Dict]: 识别的实体列表，每个实体包含name和type字段
        """
        # 实现基于LLM的实体识别
        prompt = f"""从以下文本中识别实体，只识别{self.schema.get_entity_types()}类型的实体：
文本：{text}
返回格式：[{{"name":"实体名称", "type":"实体类型"}}],
约束：
1. 只返回在Schema中定义的实体类型
2. 每个实体必须包含name和type字段
请返回符合格式的实体列表。不要输出```json这样的格式，直接按照要求的返回格式进行返回就可以。
"""
        
        try:
            entities = self.llm_client.query(prompt)
            entitles_list = list(json.loads(entities))
            if not isinstance(entitles_list, list):
                raise ValueError("实体识别结果不是一个列表")
            return entitles_list
        except Exception as e:
            self.logger.error(f"实体识别失败: {e}")
            return []
    
    def entity_standardization(self, text: str, entities: List[Dict]) -> List[Dict]:
        """
        实体标准化
        Args:
            text: 原始文本
            entities: 识别的实体列表
        Returns:
            List[Dict]: 标准化后的实体列表
        """
        # 实现基于Schema的实体标准化
        standardized = []
        for entity in entities:
            entity_type = entity["type"]
            schema_def = self.schema.get(entity_type)
            if schema_def:
                standardized.append({
                    "id": f"{entity_type}_{entity['name']}",
                    "name": entity["name"],
                    "type": entity_type,
                    "properties": self._extract_properties(text, entity, schema_def)
                })
        return standardized
    
    def relation_extraction(self, text: str, entities: List[Dict]) -> List[Dict]:
        """
        关系抽取
        Args:
            text: 原始文本
            entities: 标准化后的实体列表
        Returns:
            List[Dict]: 抽取的关系列表
        """
        # 实现基于Schema的关系抽取
        relations = []
        entity_map = {e["name"]: e for e in entities}
        
        for rel_def in self.schema.get_relations():
            prompt = f"""从文本中识别{rel_def['type']}关系：
文本：{text}
实体列表：{[e['name'] for e in entities]}
返回格式：[{{"subject":"主体", "predicate":"关系类型", "object":"客体"}}],
约束：不要输出其他任何信息，只需要返回符合格式的关系列表即可。"""
            
            try:
                extracted = self.llm_client.query(prompt)
                extracted = json.loads(extracted)
                if extracted and not isinstance(extracted, list):
                    raise ValueError("关系抽取结果不是一个列表")
                for rel in extracted:
                    if rel["subject"] in entity_map and rel["object"] in entity_map:
                        relations.append({
                            "subject": entity_map[rel["subject"]]["id"],
                            "predicate": rel["predicate"],
                            "object": entity_map[rel["object"]]["id"],
                            "properties": self._extract_relation_props(text, rel)
                        })
            except Exception as e:
                self.logger.error(f"关系抽取失败: {e}")
        
        return relations
    
    def extract_triples(self, text: str) -> Dict:
        """
        完整的三元组抽取流程
        Args:
            text: 输入文本
        Returns:
            Dict: 包含entities和relations的字典
        """
        # 1. 实体识别
        entities = self.named_entity_recognition(text)
        
        # 2. 实体标准化
        standardized_entities = self.entity_standardization(text, entities)
        
        # 3. 关系抽取
        relations = self.relation_extraction(text, standardized_entities)
        
        return {
            "entities": standardized_entities,
            "relations": relations
        }
    
    def _extract_properties(self, text: str, entity: Dict, schema_def: Dict) -> Dict:
        """抽取实体属性"""
        properties = {}
        for prop in schema_def.get("properties", []):
            # 简单实现：从文本中提取属性值
            prompt = f"""从文本中提取{entity['name']}的{prop['name']}属性值：
文本：{text}
返回格式：{{"value":"属性值"}},
约束：我不需要你输出任何其他信息，只需要返回属性值即可。"""
            try:
                result = self.llm_client.query(prompt)
                if result and "value" in result:
                    result = json.loads(result)
                    properties[prop["name"]] = result["value"]
            except:
                continue
        return properties
    
    def _extract_relation_props(self, text: str, relation: Dict) -> Dict:
        """抽取关系属性"""
        # 简单实现：记录来源文本
        return {
            "source_text": text,
            "extraction_method": "llm"
        }

class Schema:
    """简单的Schema定义类"""
    
    def __init__(self, definitions):
        self.definitions = definitions
    
    def get(self, entity_type):
        return self.definitions.get(entity_type)
    
    def get_entity_types(self):
        return list(self.definitions.keys())
    
    def get_relations(self):
        relations = []
        for entity_type, defs in self.definitions.items():
            if "relations" in defs:
                relations.extend(defs["relations"])
        return relations


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
# 使用示例
if __name__ == "__main__":
    # 示例Schema定义
    schema_def = {
        "人物": {
            "properties": [
                {"name": "职业", "type": "string"},
                {"name": "年龄", "type": "number"}
            ],
            "relations": [
                {"type": "配偶关系", "domain": "人物", "range": "人物"},
                {"type": "父子关系", "domain": "人物", "range": "人物"}
            ]
        },
        "公司": {
            "properties": [
                {"name": "行业", "type": "string"},
                {"name": "成立时间", "type": "date"}
            ],
            "relations": [
                {"type": "任职", "domain": "人物", "range": "公司"}
            ]
        }
    }
    
    
    llm = MockLLMClient()
    
    # 创建Schema和抽取器
    schema = Schema(schema_def)
    extractor = SchemaBasedExtractor(schema, llm_client=llm)  # 需要传入实际的LLM客户端
    
    # 示例文本处理
    text = "马云是阿里巴巴集团的创始人，现任董事局主席。"
    result = extractor.extract_triples(text)
    print(result)