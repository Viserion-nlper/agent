"""
示例Schema定义
"""

SCHEMA_DEFINITIONS = {
    # 人物实体定义
    "人物": {
        "properties": [
            {"name": "姓名", "type": "string", "required": True},
            {"name": "职业", "type": "string"},
            {"name": "出生日期", "type": "date"}
        ],
        "relations": [
            {
                "type": "配偶关系", 
                "domain": "人物", 
                "range": "人物",
                "properties": [
                    {"name": "结婚日期", "type": "date"}
                ]
            },
            {
                "type": "任职", 
                "domain": "人物", 
                "range": "公司",
                "properties": [
                    {"name": "职位", "type": "string"},
                    {"name": "入职时间", "type": "date"}
                ]
            }
        ]
    },
    
    # 公司实体定义
    "公司": {
        "properties": [
            {"name": "名称", "type": "string", "required": True},
            {"name": "行业", "type": "string"},
            {"name": "成立时间", "type": "date"}
        ],
        "relations": [
            {
                "type": "子公司", 
                "domain": "公司", 
                "range": "公司",
                "properties": [
                    {"name": "持股比例", "type": "float"}
                ]
            }
        ]
    },
    
    # 地点实体定义
    "地点": {
        "properties": [
            {"name": "名称", "type": "string", "required": True},
            {"name": "类型", "type": "string"},
            {"name": "坐标", "type": "geo"}
        ]
    }
}

# 预定义的关系类型
RELATION_TYPES = {
    "配偶关系": "人物-人物",
    "父子关系": "人物-人物",
    "任职": "人物-公司",
    "子公司": "公司-公司",
    "位于": "公司-地点"
}