from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import config

class CustomE5Embedding(HuggingFaceEmbeddings):
    def embed_documents(self, texts):
        texts = [f"passage: {t}" for t in texts]
        return super().embed_documents(texts)

    def embed_query(self, text):
        return super().embed_query(f"query: {text}")

embedding_model = CustomE5Embedding(model_name="intfloat/multilingual-e5-small")

db = FAISS.load_local(config.FAISS_DB_PATH, embedding_model, allow_dangerous_deserialization=True)

faiss_retriever = db.as_retriever(search_kwargs={"k": 4})

# 從 FAISS docstore 取出所有文件物件
docs = list(db.docstore._dict.values())

# 建立 BM25，同樣取 k=4
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 4

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.4, 0.6]   # BM25 佔 40%，FAISS 佔 60%
)

def extract_search_queries(learning_plan: str) -> list:
    """從 Planner 輸出中擷取 SEARCH_QUERIES 區塊"""
    if "SEARCH_QUERIES:" not in learning_plan:
        return [learning_plan[:200]]  # fallback：直接用計畫前段
    lines = learning_plan.split("SEARCH_QUERIES:")[-1].strip().splitlines()
    return [l.lstrip("- ").strip() for l in lines if l.strip().startswith("-")]

def retrieve(learning_plan: str) -> str:
    queries = extract_search_queries(learning_plan)
    all_docs = []
    seen = set()
    for q in queries:
        docs = ensemble_retriever.invoke(q)
        for doc in docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                all_docs.append(doc)
    return "\n\n".join([doc.page_content for doc in all_docs[:8]])
