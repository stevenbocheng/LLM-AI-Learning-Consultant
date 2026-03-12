from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
FAISS_DB_PATH = os.getenv("FAISS_DB_PATH", "faiss_db")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("缺少必填設定：OPENAI_API_KEY，請檢查 .env 檔案")

