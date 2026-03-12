"""
build_faiss_db.py
從 Goodreads 資料集取樣學習相關書籍，建立 FAISS 向量資料庫
"""
import os
import pandas as pd
from tqdm import tqdm
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import config

# ── CustomE5Embedding（與 retriever.py 相同）────────────────
class CustomE5Embedding(HuggingFaceEmbeddings):
    def embed_documents(self, texts):
        texts = [f"passage: {t}" for t in texts]
        return super().embed_documents(texts)
    def embed_query(self, text):
        return super().embed_query(f"query: {text}")

# ── 1. 載入原始資料 ──────────────────────────────────────────
PARQUET_PATH = r"C:\Users\b3263\.cache\kagglehub\datasets\ishanrealstate\goodreads-cleaned-dataset\versions\1\Goodreads Books\books_clean.parquet"

print("載入資料集...")
df = pd.read_parquet(PARQUET_PATH)
print(f"原始筆數：{len(df):,}")

# ── 2. 取樣策略：學習相關類型優先 ───────────────────────────
LEARNING_GENRES = [
    "science", "education", "self_help", "business", "psychology",
    "philosophy", "nonfiction", "technology", "computer", "programming",
    "mathematics", "economics", "sociology", "reference", "history",
    "biography", "memoir", "health"
]

def has_learning_genre(genres_val):
    if not isinstance(genres_val, (list, str)):
        return False
    genres_str = str(genres_val).lower()
    return any(g in genres_str for g in LEARNING_GENRES)

# 過濾：有摘要 + 摘要夠長
df_valid = df[
    df["summary_clean"].notna() &
    (df["summary_clean"].str.len() > 150)
].copy()

# 分兩批取樣
df_learning = df_valid[df_valid["genres"].apply(has_learning_genre)]
df_learning_top = df_learning.sort_values("star_rating", ascending=False).head(3000)

df_general = df_valid.sort_values(
    ["num_ratings", "star_rating"], ascending=False
).head(2000)

df_sample = pd.concat([df_learning_top, df_general]).drop_duplicates(subset=["id"])
print(f"取樣後筆數：{len(df_sample):,}（學習類 {len(df_learning_top):,} + 高評分 {len(df_general):,}）")

# ── 3. 轉換為 LangChain Document ────────────────────────────
def row_to_document(row):
    author = row["author"]
    if isinstance(author, list):
        author = ", ".join(author)

    genres = row["genres"]
    if isinstance(genres, list):
        genres = ", ".join(genres[:5])  # 最多顯示 5 個類型

    content = (
        f"Title: {row['name']}\n"
        f"Author: {author}\n"
        f"Genres: {genres}\n"
        f"Rating: {row['star_rating']:.2f} ({int(row['num_ratings']):,} ratings)\n"
        f"Published: {row.get('pub_year', 'N/A')}\n\n"
        f"{row['summary_clean']}"
    )
    return Document(
        page_content=content,
        metadata={"title": row["name"], "author": str(author), "source": "goodreads"}
    )

print("轉換文件格式...")
documents = [row_to_document(row) for _, row in tqdm(df_sample.iterrows(), total=len(df_sample))]

# ── 4. 切分文字 ──────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
split_docs = splitter.split_documents(documents)
print(f"切分後 chunk 數：{len(split_docs):,}")

# ── 5. 建立向量索引（分批）──────────────────────────────────
print("載入 Embedding 模型...")
embedding_model = CustomE5Embedding(model_name="intfloat/multilingual-e5-small")

batch_size = 2000
vectorstore = None

print("開始建立向量索引...")
for i in tqdm(range(0, len(split_docs), batch_size), desc="分批處理"):
    batch = split_docs[i:i + batch_size]
    if not batch:
        continue
    if vectorstore is None:
        vectorstore = FAISS.from_documents(batch, embedding_model)
    else:
        vectorstore.add_documents(batch)

# ── 6. 儲存 ─────────────────────────────────────────────────
vectorstore.save_local(config.FAISS_DB_PATH)
print(f"\n✅ 向量資料庫已儲存至 {config.FAISS_DB_PATH}/")

# ── 7. 驗證 ─────────────────────────────────────────────────
print("\n驗證檢索結果：")
test_queries = [
    "Python programming for beginners",
    "machine learning deep learning",
    "self improvement productivity"
]
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
for q in test_queries:
    print(f"\n查詢：{q}")
    results = retriever.invoke(q)
    for r in results:
        title_line = r.page_content.split("\n")[0]
        print(f"  → {title_line}")
