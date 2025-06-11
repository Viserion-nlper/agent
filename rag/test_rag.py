

from typing import List
from Embedding import Qwen3Embedding, ZhipuEmbedding
from VectorStore import VectorStore
from Qwen3_ReRank import Qwen3Reranker
from utils import ReadFiles


def get_top_k_similar(
    query: str, 
    embedding_model, 
    vector_db, 
    top_k: int = 5
) -> List[str]:
    """
    获取与查询最相似的前top_k个内容
    :param query: 查询文本
    :param embedding_model: 嵌入模型实例
    :param vector_db: 向量数据库实例
    :param top_k: 返回结果数量
    :return: 相似内容列表
    """
    # 执行向量查询并取前top_k个结果
    results = vector_db.query(
        query, 
        EmbeddingModel=embedding_model, 
        k=top_k
    )
    return [result for result in results[:top_k]]




docs = ReadFiles('./data/test_rag_data').get_content(max_token_len=50, cover_content=30) # 获得data目录下的所有文件内容并分割
vector = VectorStore(docs)
# embedding = ZhipuEmbedding() # 创建EmbeddingModel
embedding = Qwen3Embedding()

vector.get_vector(EmbeddingModel=embedding)
vector.persist(path='./data/test_rag_data') # 将向量和文档内容保存到项目目录下，下次再用就可以直接加载本地的数据库


vector = VectorStore()
vector.load_vector('./data/test_rag_data') # 加载本地的知识库

print(len(vector.vectors[0]))

# embedding = ZhipuEmbedding()# 创建EmbeddingModel
question = '农村村民建住宅，需符合什么？'

top_k = 20  # 设定需要获取的相似结果数量
contents = get_top_k_similar(question, embedding, vector, top_k)

## 再进行rerank模型查询
model = Qwen3Reranker(model_name_or_path='Qwen/Qwen3-Reranker-0.6B', instruction="Retrieval document that can answer user's query", max_length=2048)

pairs = list(zip(question, contents))
instruction="Given the user query, retrieval the relevant passages"
new_scores = model.compute_scores(pairs, instruction)
print('scores', new_scores)
print(f'知识库输出：{contents}')
# TODO: 这里进行了Embedding和Rerank模型的查询，后续可以将结果进行处理，返回给用户，后面接入大模型总结回复。