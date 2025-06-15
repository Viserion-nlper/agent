# 基于Schema的三元组抽取系统

## 功能概述
本系统提供从非结构化文本中抽取符合预定义Schema的三元组(实体-关系-实体)的能力，主要包含以下功能：
- 命名实体识别
- 实体标准化
- 关系抽取 
- 子图构建

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 定义Schema
在`schema_definitions.py`中定义您的领域Schema，示例：
```python
SCHEMA = {
    "人物": {
        "properties": [...],
        "relations": [...]
    },
    "公司": {...}
}
```

### 3. 初始化抽取器
```python
from schema_based_extractor import SchemaBasedExtractor
from example_schema import SCHEMA_DEFINITIONS

# 初始化LLM客户端 (需自行实现)
llm_client = YourLLMClient() 

# 创建抽取器
extractor = SchemaBasedExtractor(
    schema=SCHEMA_DEFINITIONS,
    llm_client=llm_client
)
```

### 4. 执行抽取
```python
text = "马云是阿里巴巴集团的创始人"
result = extractor.extract_triples(text)

# 输出结果
print("识别实体:", result["entities"]) 
print("识别关系:", result["relations"])
```

## 核心接口说明

### SchemaBasedExtractor类
- `named_entity_recognition(text: str)`: 识别文本中的实体
- `entity_standardization(text, entities)`: 标准化识别出的实体
- `relation_extraction(text, entities)`: 抽取实体间关系
- `extract_triples(text)`: 完整的三元组抽取流程

## 高级配置

### 自定义抽取策略
继承`SchemaBasedExtractor`类并重写相关方法：
```python
class CustomExtractor(SchemaBasedExtractor):
    def named_entity_recognition(self, text):
        # 自定义实体识别逻辑
        ...
```

### 性能优化建议
1. 批量处理文本提高效率
2. 对长文本先进行分块处理
3. 缓存频繁出现的实体识别结果
