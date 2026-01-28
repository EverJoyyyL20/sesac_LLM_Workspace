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
from langraph_setting import app


def run_langgraph(question: str, messages=None):

    if messages is None:
        messages = []

    state = {
        "question": question,
        "messages": messages
    }

    result = app.invoke(state)

    return result["answer"], result.get("messages", [])


if __name__ == "__main__":

    history = []

    print("=== LangGraph Chatbot 시작 ===")

    while True:
        user_input = input("\n사용자: ")

        if user_input.lower() in ["exit", "quit"]:
            print("종료합니다.")
            break

        answer, history = run_langgraph(user_input, history)

        print("\n봇:", answer)
