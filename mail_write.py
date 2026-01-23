import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="이메일 작성 서비스예요~",page_icon=":robot:")
st.header("이메일 작성기")

def getEmail():
    input_text=st.text_area(label="메일 입력",
                            label_visibility='collapsed',
                            placeholder="당신의 메일은..."
                            ,key="input_text")
    return input_text

input_text=getEmail()

query_template="""
메일을 작성해주세요.
아래는 이메일입니다.:
이메일:{email}
"""

from langchain_classic import PromptTemplate

prompt=PromptTemplate(
    input_variables=["email"],
    template=query_template,
)

from langchain_classic.chat_models import ChatOpenAI

def loadLanguageModel():
    llm=ChatOpenAI(temperature=0.0,model_name='gpt-4.1-mini')
    return llm

st.button("*예제를 보여주세요*",type='secondary',help="붓이 작성한 메일을 확인해보세요.")
st.markdown("### 봇이 작성한 메일은:")

if input_text:
    llm=loadLanguageModel()
    prompt_with_email=prompt.format(email=input_text)
    formatted_email=llm.invoke(prompt_with_email).content
    st.write(formatted_email)