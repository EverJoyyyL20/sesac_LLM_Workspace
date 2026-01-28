# app.py
import streamlit as st
from dotenv import load_dotenv
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

from main import run_langgraph

load_dotenv(override=True)

st.set_page_config(page_title="5대 법령 비서", page_icon="⚖️")
st.title("⚖️ 5대 법령 전문 AI 비서")
st.caption("개인정보보호법 | 근로기준법 | 도로교통법 | 전자상거래법 | 주택임대차보호법")

# Streamlit 전용 메모리
msgs = StreamlitChatMessageHistory(key="legal_chat_history")

# 기존 대화 출력
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# BaseMessage -> dict 변환
def convert_to_dict_history(messages):
    history = []
    for m in messages:
        history.append({
            "role": "user" if m.type == "human" else "assistant",
            "content": m.content
        })
    return history


if user_input := st.chat_input("궁금한 법령에 대해 물어보세요."):

    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):

        placeholder = st.empty()
        full_response = ""

        # LangGraph용 history 변환
        graph_history = convert_to_dict_history(msgs.messages)

        # Graph 실행
        answer, updated_history = run_langgraph(
            question=user_input,
            messages=graph_history
        )

        # 스트리밍 효과
        for ch in answer:
            full_response += ch
            placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)

        # Streamlit 메모리에 저장 (Graph 결과 재사용 ❌)
        msgs.add_user_message(user_input)
        msgs.add_ai_message(answer)
