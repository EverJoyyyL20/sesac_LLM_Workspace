import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

import streamlit as st
from langchain_classic.chat_models import ChatOpenAI
st.set_page_config(page_title="🦜🔗 뭐든지 질문하세요~ ")
st.title('🦜🔗 뭐든지 질문하세요~ ')

def generate_response(input_text):  #llm이 답변 생성
    llm = ChatOpenAI(temperature=0,  # 창의성 0으로 설정 
                 model_name='gpt-4.1-mini',  # 모델명
                )
    st.info(llm.invoke(input_text).content)

with st.form('Question'):
    text = st.text_area('질문 입력:', 'What types of text models does OpenAI provide?') #첫 페이지가 실행될 때 보여줄 질문
    submitted = st.form_submit_button('보내기')
    generate_response(text)