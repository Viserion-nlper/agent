from glob import glob
import json
text_lines = []
for file_path in glob("./data/milvus_docs/en/faq/*.md", recursive=True):
    with open(file_path, "r") as file:        
        file_text = file.read()    
        text_lines += file_text.split("# ")
        
        
import requests
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

embedding_model = SentenceTransformer("/Users/mac/.cache/modelscope/hub/models/Qwen/Qwen3-Embedding-0.6B")

reranker_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B", padding_side='left')
reranker_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B", trust_remote_code=True, torch_dtype=torch.float16).to('mps').eval()

# Reranker 配置
token_false_id = reranker_tokenizer.convert_tokens_to_ids("no")
token_true_id = reranker_tokenizer.convert_tokens_to_ids("yes")
max_reranker_length = 8192
prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
prefix_tokens = reranker_tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = reranker_tokenizer.encode(suffix, add_special_tokens=False)

def emb_text(text, is_query=False):
    """    使用Qwen3-Embedded-0.6B模型来生成文本嵌入.    
    Args:        
    text: Input text to embed        
    is_query: 区分是query语句还是document内容   
    Returns:   返回文本的嵌入向量列表    
    """    
    if is_query:        
        # For queries, use the "query" prompt for better retrieval performance        
        embeddings = embedding_model.encode([text], prompt_name="query")    
    else:        # For documents, use default encoding        
        embeddings = embedding_model.encode([text])    
    return embeddings[0].tolist()

#然后再定义重排序函数来提升检索的质量，构建pipline
def format_instruction(instruction, query, doc): 
    """Format instruction for reranker input"""    
    if instruction is None:
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'    
    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
            instruction=instruction, query=query, doc=doc    
        )    
    return output

def process_inputs(pairs): 
    """Process inputs for reranker"""    
    inputs = reranker_tokenizer(
        pairs, padding=False, truncation='longest_first',        
        return_attention_mask=False, max_length=max_reranker_length - len(prefix_tokens) - len(suffix_tokens)    
        )    
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens    
    inputs = reranker_tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_reranker_length)    
    for key in inputs:        
        inputs[key] = inputs[key].to(reranker_model.device)    
    return inputs

@torch.no_grad()
def compute_logits(inputs, **kwargs):
    """Compute relevance scores using reranker"""
    batch_scores = reranker_model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()
    return scores

def rerank_documents(query, documents, task_instruction=None):
    """
    Rerank documents based on query relevance using Qwen3-Reranker-0.6B model.
    Args:
        query: Search query
        documents: List of documents to rerank
        task_instruction: Task instruction for reranking
    Returns:
        List of (document, score) tuples sorted by relevance score
    """
    if task_instruction is None:
        task_instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    # Format inputs for reranker
    pairs = [format_instruction(task_instruction, query, doc) for doc in documents]
    # Process inputs and compute scores
    inputs = process_inputs(pairs)
    scores = compute_logits(inputs)
    # Combine documents with scores and sort by score (descending)
    doc_scores = list(zip(documents, scores))
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    return doc_scores


# 简单生成一个测试向量，并打印其维度和前几个元素
test_embedding = emb_text("This is a test")
embedding_dim = len(test_embedding)
print(f"Embedding dimension: {embedding_dim}")
print(f"First few elements of the embedding: {test_embedding[:10]}")

def query_with_rag(milvus_client, collection_name, question):
    """
    执行完整的RAG查询流程，包括：
    1. 使用Milvus进行Embedding初步粗排检索
    2. 使用Qwen3-Reranker进行重排序
    3. 使用Qwen2-LLM生成最终回答
    
    参数:
        milvus_client: Milvus客户端实例
        collection_name: Milvus集合名称
        question: 查询问题
    
    返回:
        dict: 包含检索结果和生成回答的字典
    """
    # Step 1: Initial retrieval with larger candidate set
    search_res = milvus_client.search(
        collection_name=collection_name,    
        data=[emb_text(question, is_query=True)],    
        limit=10,
        search_params={"metric_type": "IP", "params": {}},
        output_fields=["text"],
    )

    # Step 2: Extract candidate documents for reranking
    candidate_docs = [res["entity"]["text"] for res in search_res[0]]
    
    # Step 3: Rerank documents using Qwen3-Reranker
    print("Reranking documents...")
    reranked_docs = rerank_documents(question, candidate_docs)
    
    # Step 4: Select top 3 reranked documents
    top_reranked_docs = reranked_docs[:3]
    print(f"Selected top {len(top_reranked_docs)} documents after reranking")

    # Display reranked results with reranker scores
    reranked_lines_with_scores = [(doc, score) for doc, score in top_reranked_docs]
    print("Reranked results:")
    print(json.dumps(reranked_lines_with_scores, indent=4))
    
    # Also show original embedding-based results for comparison
    print("\n" + "="*80)
    print("Original embedding-based results (top 3):")
    original_lines_with_distances = [
        (res["entity"]["text"], res["distance"]) for res in search_res[0][:3]
    ]
    print(json.dumps(original_lines_with_distances, indent=4, ensure_ascii=False))

    # Prepare context for final answer generation
    context = "\n".join([line_with_distance[0] for line_with_distance in original_lines_with_distances])
    
    # Generate final answer using Qwen2
    SYSTEM_PROMPT = """
    Human: You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided.
    """
    USER_PROMPT = f"""
    Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
    <context>
    {context}
    </context>
    <question>
    {question}
    """
    
    analysis = {
            "model": "qwen2",
            "prompt": USER_PROMPT,
            "temperature": 0.7,
            "stream": False
        }
    response = requests.post(
                'http://localhost:11434/api/generate',
                json=analysis
            )
    print(json.loads(response.text).get("response", ""))
    return json.loads(response.text).get("response", "")


collection_name = "my_rag_collection"
from pymilvus import MilvusClient
milvus_client = MilvusClient(uri="./milvus_demo.db")

if milvus_client.has_collection(collection_name):
    milvus_client.drop_collection(collection_name)
    
milvus_client.create_collection(
    collection_name=collection_name,
    dimension=embedding_dim,
    metric_type="IP",  # Inner product 计算距离    
    consistency_level="Strong",  # Strong consistency level
)

from tqdm import tqdm
data = []
for i, line in enumerate(tqdm(text_lines, desc="Creating embeddings")):
    data.append({"id": i, "vector": emb_text(line), "text": line})
milvus_client.insert(collection_name=collection_name, data=data)

# 示例调用
if __name__ == "__main__":
    result = query_with_rag(milvus_client, collection_name, "该项目的预算情况如何?")
    print("\n最终结果:")
    print(result["answer"])