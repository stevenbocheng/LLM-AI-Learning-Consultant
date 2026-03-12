import streamlit as st
import agents
import config
from graph import advisor_graph
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

st.set_page_config(page_title="AI 學習諮詢師", page_icon="📚", layout="wide")

st.title("AI 學習諮詢師")
st.caption("輸入你的學習背景與目標，系統將為你規劃個人化學習路線並推薦適合的書籍資源。")

# ── 側邊欄設定 ───────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 設定")

    user_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="輸入你的 OpenAI API Key，不會被儲存。"
    )

    if user_api_key:
        st.success("API Key 已輸入")
    else:
        st.info("請輸入 OpenAI API Key 才能使用")

    st.markdown("---")
    st.markdown("**流程說明**")
    st.markdown("""
1. **Planner** — 分析背景，規劃學習路線
2. **Retriever** — 從書籍資料庫檢索相關資源
3. **Writer** — 撰寫學習建議文案
""")
    st.markdown("---")
    st.caption("資料來源：Kaggle Goodreads Dataset")

# ── 取得有效的 API Key ───────────────────────────────────────
def get_api_key() -> str:
    return user_api_key.strip()

def is_learning_query(text: str) -> bool:
    llm = ChatOpenAI(model=config.OPENAI_MODEL, api_key=get_api_key(), max_tokens=10)
    response = llm.invoke([
        SystemMessage(content="你是一個分類器。判斷使用者輸入是否與學習、職涯發展、技能提升、讀書計畫等學習諮詢相關。只回答 YES 或 NO，不要說其他任何話。"),
        HumanMessage(content=text)
    ])
    return response.content.strip().upper().startswith("YES")

# ── 輸入區 ───────────────────────────────────────────────────
user_input = st.text_area(
    "請描述你的背景與學習目標",
    placeholder="例如：我是大學二年級學生，學過基礎 Python，想往機器學習方向發展，目標是一年內能看懂論文並實作模型，每週可投入約 10 小時。",
    height=120
)

run_btn = st.button("開始規劃", type="primary", use_container_width=True)

# ── 執行與顯示 ───────────────────────────────────────────────
if run_btn:
    if not user_input.strip():
        st.warning("請先輸入你的學習背景與目標。")
    elif not get_api_key():
        st.error("請先在左側輸入 OpenAI API Key。")
    else:
        # 動態更新 agents.py 的 LLM 實例（使用使用者輸入的 Key）
        agents.llm = ChatOpenAI(model=config.OPENAI_MODEL, api_key=get_api_key())

        if not is_learning_query(user_input):
            st.error("⚠️ 你的問題似乎與學習諮詢無關。\n\n本系統專門協助規劃學習路線與推薦學習資源，請描述你的學習背景與目標，例如：「我想學習機器學習，目前有基礎 Python 能力...」")
        else:
            with st.status("處理中...", expanded=True) as status:
                st.write("🧠 Planner 分析背景，規劃學習路線...")
                st.write("🔍 Retriever 從書籍資料庫檢索相關資源...")
                st.write("✍️ Writer 撰寫學習建議文案...")

                result = advisor_graph.invoke({
                    "user_input": user_input,
                    "learning_plan": "",
                    "rag_context": "",
                    "initial_draft": "",
                    "review_feedback": "",
                    "final_output": "",
                    "revision_count": 0
                })

                status.update(label="✅ 完成！", state="complete")

            # 最終建議
            st.markdown("---")
            st.subheader("📋 最終學習建議")
            st.markdown(result.get("final_output", ""))

            # 中間步驟（Tab）
            st.markdown("---")
            st.subheader("🔬 中間步驟詳情")
            tab1, tab2, tab3 = st.tabs(["🧠 學習計畫", "🔍 檢索結果", "✍️ 學習建議初稿"])

            with tab1:
                st.text_area("Planner 輸出", result.get("learning_plan", ""), height=300, disabled=True)
            with tab2:
                st.text_area("RAG 檢索到的書籍", result.get("rag_context", ""), height=300, disabled=True)
            with tab3:
                st.text_area("Writer 初稿", result.get("initial_draft", ""), height=300, disabled=True)
