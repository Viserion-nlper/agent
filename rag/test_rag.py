

from Embedding import ZhipuEmbedding
from VectorStore import VectorStore
from utils import ReadFiles

docs = ReadFiles('./data/test_rag_data').get_content(max_token_len=50, cover_content=30) # 获得data目录下的所有文件内容并分割
vector = VectorStore(docs)
embedding = ZhipuEmbedding() # 创建EmbeddingModel
vector.get_vector(EmbeddingModel=embedding)
vector.persist(path='./data/test_rag_data') # 将向量和文档内容保存到项目目录下，下次再用就可以直接加载本地的数据库


vector = VectorStore()
vector.load_vector('./data/test_rag_data') # 加载本地的知识库

print(len(vector.vectors[0]))

embedding = ZhipuEmbedding()# 创建EmbeddingModel
question = '归因分析引擎青源峰达工业赋能大模型可行性方案项目背景与目标是什么?'
content = vector.query(question, EmbeddingModel=embedding, k=2)[0]
print(f'知识库输出：{content}')