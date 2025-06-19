# 基于Schema的知识图谱构建系统 (KAG)

## 项目概述
KAG(Knowledge Acquisition Graph)是一个基于Schema的三元组抽取和知识图谱构建系统，主要功能包括：
- 从非结构化文本中抽取符合预定义Schema的三元组(实体-关系-实体)
- 实体标准化和关系验证
- 知识图谱构建和存储
- 支持自定义Schema和扩展

## 核心功能
1. **Schema定义**：支持灵活定义实体类型、属性和关系
2. **三元组抽取**：
   - 命名实体识别
   - 实体标准化
   - 关系抽取
3. **向量化服务**：支持文本向量化
4. **知识图谱构建**：支持将抽取结果构建为知识图谱

## 架构设计
```
├── schema_based_extractor.py  # 核心抽取逻辑
├── common/
│   ├── llm/                  # LLM客户端实现
│   └── vectorize_model/      # 向量化模型
├── templates/               # 项目模板
├── examples/                 # 示例配置和Schema
└── configs/                 # 配置文件
```

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 定义Schema
在`schema_definitions.py`中定义您的领域Schema，示例：
```python
SCHEMA_DEFINITIONS = {
    "人物": {
        "properties": [
            {"name": "姓名", "type": "string", "required": True}
        ],
        "relations": [
            {"type": "任职", "domain": "人物", "range": "公司"}
        ]
    }
}
```

### 3. 配置系统
编辑`zhiqiang_config.yaml`：
```yaml
llm:
  type: mock  # 使用模拟LLM

vectorize_model:
  type: mock
  vector_dimensions: 768
```

### 4. 运行抽取
```python
from schema_based_extractor import SchemaBasedExtractor
from example_schema import SCHEMA_DEFINITIONS

extractor = SchemaBasedExtractor(
    schema=SCHEMA_DEFINITIONS,
    llm_client=YourLLMClient()
)

text = "马云是阿里巴巴集团的创始人"
result = extractor.extract_triples(text)
```

## 配置说明
- **LLM配置**：支持OpenAI等LLM服务
- **向量化配置**：支持多种向量化模型
- **日志配置**：可调整日志级别

## 高级用法
1. **自定义抽取器**：继承`SchemaBasedExtractor`类
2. **批量处理**：支持批量文本处理
3. **性能优化**：缓存、并行处理等

## 示例
见`examples/`目录下的示例配置和Schema定义