from langgraph.graph import StateGraph, END
from langraph_node import (
    ChatState,
    classify_node,
    retrieve_node,
    law_answer_node,
    simple_answer_node,
    impossible_answer_node,
    router
)

from langgraph.graph import StateGraph, END

workflow = StateGraph(ChatState)

# 노드 등록
workflow.add_node("classify", classify_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("law_answer", law_answer_node)
workflow.add_node("simple_answer", simple_answer_node)
workflow.add_node("impossible_answer", impossible_answer_node)

# 시작점
workflow.set_entry_point("classify")

# 핵심: 조건 분기
workflow.add_conditional_edges(
    "classify",
    router,
    {
        "law": "retrieve",
        "simple": "simple_answer",
        "impossible": "impossible_answer"
    }
)

# retrieve는 law만 통과
workflow.add_edge("retrieve", "law_answer")

# 종료
workflow.add_edge("law_answer", END)
workflow.add_edge("simple_answer", END)
workflow.add_edge("impossible_answer", END)

app = workflow.compile()


# result = app.invoke({
#     "question": "나 짤렸어",
#     "messages": []
# })

# print(result["answer"])
