from rag2 import (
    basic_chain_setting,
    classify_chain_setting,
    simple_chain_setting,
    impossible_chain_setting,
)
from retriever import load_chroma_similarity_retriever   # 너가 만든 RAG 검색 함수라고 가정
from typing import TypedDict, List, Dict
from langchain_core.documents import Document

# 체인 생성
classify_chain = classify_chain_setting()
law_chain = basic_chain_setting()
simple_chain = simple_chain_setting()
impossible_chain = impossible_chain_setting()

retriever = load_chroma_similarity_retriever(
    chroma_path="./chroma_law_db_bge"
)
class ChatState(TypedDict, total=False):
    question: str
    answer: str
    classify: str
    search_results: List[Document]
    messages: List[Dict[str, str]]  



def classify_node(state: ChatState) -> ChatState:

    category = classify_chain.invoke({
        "question": state["question"]
    }).strip()

    return {
        **state,
        "classify": category
    }


def router(state: ChatState) -> str:
    return state.get("classify", "").strip().lower()



def retrieve_node(state: ChatState) -> ChatState:

    question = state.get("question", "")

    if not question:
        return state

    docs = retriever.invoke(question)

    return {
        **state,
        "search_results": docs
    }



def law_answer_node(state: ChatState) -> ChatState:

    context_text = "\n\n".join(
        [doc.page_content for doc in state["search_results"]]
    )

    answer = law_chain.invoke({
        "question": state["question"],
        "context_text": context_text
    })

    messages = state.get("messages", [])

    messages.append({
        "role": "user",
        "content": state["question"]
    })

    messages.append({
        "role": "assistant",
        "content": answer
    })

    return {
        **state,
        "answer": answer,
        "messages": messages
    }

def simple_answer_node(state: ChatState) -> ChatState:

    answer = simple_chain.invoke({
        "question": state["question"],
        "context": ""
    })

    messages = state.get("messages", [])

    messages.append({"role": "user", "content": state["question"]})
    messages.append({"role": "assistant", "content": answer})

    return {
        **state,
        "answer": answer,
        "messages": messages
    }



def impossible_answer_node(state: ChatState) -> ChatState:

    answer = impossible_chain.invoke({
        "question": state["question"],
        "context": ""
    })

    messages = state.get("messages", [])

    messages.append({
        "role": "user",
        "content": state["question"]
    })

    messages.append({
        "role": "assistant",
        "content": answer
    })

    return {
        **state,
        "answer": answer,
        "messages": messages
    }
