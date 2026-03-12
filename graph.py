from typing import TypedDict
from agents import run_planner, run_writer  # , run_reviewer  # 暫時停用 Reviewer
from langgraph.graph import StateGraph, END
from retriever import retrieve

class AdvisorState(TypedDict):
    user_input: str       # 使用者輸入（一開始放進去，不會改變）
    learning_plan: str    # Planner 寫進來
    rag_context: str      # Retriever 寫進來
    initial_draft: str    # Writer 寫進來
    review_feedback: str  # Reviewer 寫進來
    final_output: str     # Writer 最終版寫進來
    revision_count: int   # 每次重寫加 1，上限 2

def plan_node(state: AdvisorState) -> dict:
    plan = run_planner(state["user_input"])
    return {"learning_plan": plan}

def retrieve_node(state: AdvisorState) -> dict:
    context = retrieve(state["learning_plan"])  # 不變，已經傳 learning_plan
    return {"rag_context": context}

def write_node(state: AdvisorState) -> dict:
    draft = run_writer(
        state["learning_plan"],
        state["rag_context"],
        state.get("review_feedback", "")
    )
    # 每次寫完都同時更新 initial_draft 和 final_output
    return {
        "initial_draft": draft,
        "final_output": draft,
        "revision_count": state.get("revision_count", 0) + 1
    }

def review_node(state: AdvisorState) -> dict:
    feedback = run_reviewer(state["initial_draft"], state["user_input"])
    return {"review_feedback": feedback}

# 建立工廠（圖）
builder = StateGraph(AdvisorState)

# 把工人加進工廠
builder.add_node("planner", plan_node)
builder.add_node("retriever", retrieve_node)
builder.add_node("writer", write_node)
# builder.add_node("reviewer", review_node)  # 暫時停用

# 定義傳送帶順序：Planner → Retriever → Writer → END
builder.set_entry_point("planner")
builder.add_edge("planner", "retriever")
builder.add_edge("retriever", "writer")
builder.add_edge("writer", END)

# 暫時停用 Reviewer 與條件分支
# builder.add_edge("writer", "reviewer")
# def should_revise(state):
#     if '{"pass": false}' in state.get("review_feedback","") and state.get("revision_count",0) < 2:
#         return "revise"
#     return "end"
# builder.add_conditional_edges("reviewer", should_revise, {"revise": "writer", "end": END})

# 工廠組裝完成，可以開工了
advisor_graph = builder.compile()