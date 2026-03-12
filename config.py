from dotenv import load_dotenv
import os

load_dotenv()

def _get(key, default=None):
    """優先讀取環境變數，再嘗試 Streamlit Secrets（雲端部署用）"""
    value = os.getenv(key, default)
    if not value:
        try:
            import streamlit as st
            value = st.secrets.get(key, default)
        except Exception:
            pass
    return value

OPENAI_API_KEY = _get("OPENAI_API_KEY")
OPENAI_MODEL = _get("OPENAI_MODEL", "gpt-4o")
FAISS_DB_PATH = _get("FAISS_DB_PATH", "faiss_db")
LANGCHAIN_API_KEY = _get("LANGCHAIN_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("缺少必填設定：OPENAI_API_KEY，請檢查 .env 檔案或 Streamlit Secrets")

