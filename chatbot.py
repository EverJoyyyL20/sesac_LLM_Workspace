#!pip install streamlit-chat streamlit langchain faiss-cpu



# ==============================
# 운영체제 환경변수 접근용 모듈
# ==============================
import os

# ==============================
# .env 파일을 읽기 위한 라이브러리
# ==============================
from dotenv import load_dotenv

# ==============================
# .env 파일 안의 환경변수 불러오기
# ==============================
load_dotenv()

# ==============================
# OPENAI_API_KEY 값을 환경변수에서 가져오기
# (.env 파일 안에 저장되어 있음)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ==============================
# Streamlit 웹 UI 라이브러리
# ==============================
import streamlit as st

# ==============================
# 채팅 형태 말풍선 UI 출력용 라이브러리
# ==============================
from streamlit_chat import message

# ==============================
# OpenAI 임베딩 모델
# 텍스트 → 숫자 벡터로 변환하는 역할
# ==============================
from langchain_classic.embeddings.openai import OpenAIEmbeddings


# ==============================
# OpenAI GPT 채팅 모델
# ==============================
from langchain_classic.chat_models import ChatOpenAI

# ==============================
# "문서 검색 + 대화형 챗봇" 체인
# ==============================
from langchain_classic.chains import ConversationalRetrievalChain

# ==============================
# FAISS 벡터 데이터베이스
# (문서 벡터 저장 + 유사도 검색)
# ==============================

from langchain_classic.vectorstores import FAISS

# ==============================
# 임시 파일 생성용 모듈
# ==============================
import tempfile

# ==============================
# PDF 파일을 읽어서 텍스트로 변환하는 로더
# ==============================
from langchain_classic.document_loaders import PyPDFLoader


# ==================================================
# Streamlit 사이드바에 PDF 업로드 버튼 생성
# ==================================================
uploaded_file = st.sidebar.file_uploader("upload", # 버튼 이름
                                         type="pdf")# PDF 파일만 허용


# ==================================================
# 파일이 업로드 되었을 때만 실행
# ==================================================
if uploaded_file :
     # ==========================================
    # 업로드된 PDF는 메모리에만 존재함
    # → 실제 파일 형태로 임시 저장 필요
    # ==========================================
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        # 업로드된 PDF 파일의 바이너리 데이터를
        # 임시 파일에 그대로 저장
        tmp_file.write(uploaded_file.getvalue())
        # 임시 파일 경로 저장
        tmp_file_path = tmp_file.name

    # ==========================================
    # PDF 파일 로드 객체 생성
    # ==========================================
    loader = PyPDFLoader(tmp_file_path)

    # ==========================================
    # PDF 내용을 텍스트(Document 객체)로 변환
    # ==========================================
    data = loader.load()

    # ==========================================
    # OpenAI 임베딩 모델 생성
    # ==========================================
    embeddings = OpenAIEmbeddings()

    # ==========================================
    # PDF 문서를 벡터 DB(FAISS)로 변환
    # ==========================================
    vectors = FAISS.from_documents(data, # PDF에서 추출한 문서들
                                   embeddings) # 벡터 변환 모델

    
    # ==========================================
    # 문서 검색 + GPT 대화 체인 생성
    # ==========================================

    # ------------------------------
    # GPT 모델 설정
    # ------------------------------                                 # 창의성 최소 (정확도 중시)
                                                                    # 사용할 GPT 모델
    chain = ConversationalRetrievalChain.from_llm(llm = ChatOpenAI(temperature=0.0,model_name='gpt-4.1-mini'),
                                                  
                                                # ------------------------------
                                                # FAISS 검색기 연결
                                                # ------------------------------ 
                                                  retriever=vectors.as_retriever())


    # 사용자의 질문(query)을 받아서
    # 이전 대화 기록을 함께 LLM에게 전달하고
    # 답변을 받아 다시 대화 기록에 저장하는 함수
    def conversational_chat(query): 
        # chain에 입력값 전달
        # question : 현재 사용자가 입력한 질문
        # chat_history : 이전 대화 목록 (문맥 유지 목적)      
        result = chain({"question": query, "chat_history": st.session_state['history']})

        # 방금 대화 내용을 history 리스트에 저장
        # (질문, 답변) 형태로 저장하여 LangChain 문맥 유지에 사용
        st.session_state['history'].append((query, result["answer"]))
        # LLM이 생성한 답변만 반환
        # → 화면 출력용      
        return result["answer"]
    
# ----------------------------------------
# Streamlit 세션 상태 초기화 부분
# (새로고침 시에도 데이터 유지 목적)
# ----------------------------------------


    # history 키가 session_state에 없으면
    # 빈 리스트로 초기화
    # → LangChain 문맥 전달용 대화 저장 공간
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    # generated 키가 없으면
    # 챗봇이 처음 보여줄 인사 메시지 저장
    # uploaded_file.name → 업로드한 파일 이름 표시
    # → "안녕하세요! 파일명 에 관해 질문주세요."
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["안녕하세요! " + uploaded_file.name + "에 관해 질문주세요."]

    # past 키가 없으면
    # 사용자가 입력한 메시지 기록 저장용 리스트 생성
    # 채팅 UI에서 사용자 말풍선 역할
    if 'past' not in st.session_state:
        st.session_state['past'] = ["안녕하세요!"]
    
    # ----------------------------------------
    # 화면 UI 레이아웃 구성
    # ----------------------------------------


    # 챗봇 응답(대화 이력)을 출력할 영역 생성
    # 이 컨테이너 안에 채팅 로그가 표시됨
    #챗봇 이력에 대한 컨테이너
    response_container = st.container()
    
    # 사용자가 질문을 입력하는 영역 생성
    # input box, 버튼 등이 들어갈 공간
    container = st.container()

    # ---------------------------------------------
    # 사용자 입력 영역 (질문 입력 + 전송 버튼 처리)
    # ---------------------------------------------

    # 이전에 생성한 container 영역 안에 UI 구성
    # → 사용자가 질문을 입력하는 구역
    with container: #대화 내용 저장(기억)
        # 입력값을 한번에 제출하기 위한 Form 생성
        # key: Streamlit 내부에서 구분하기 위한 고유 ID
        # clear_on_submit=True: Send 버튼 누르면 입력창 자동 초기화
        with st.form(key='Conv_Question', clear_on_submit=True):
            # 텍스트 입력창 생성
            # "Query:" → 입력창 위에 표시될 라벨
            # placeholder → 입력 전에 보여줄 안내 문구
            # key → Streamlit 상태 관리용 고유 이름        
            user_input = st.text_input("Query:", placeholder="PDF파일에 대해 얘기해볼까요? (:", key='input')
            # 전송 버튼 생성
        #    버튼 클릭 시 submit_button 값이 True로 변경됨
            submit_button = st.form_submit_button(label='Send')
         # -------------------------------------------------
    # 사용자가 Send 버튼을 눌렀고
    # 입력값이 비어있지 않을 때만 실행
    # -------------------------------------------------
        if submit_button and user_input:
             # LangChain 챗봇 함수 호출
            # → 질문 전달
            # → 이전 대화 포함
            # → 답변 생성
            output = conversational_chat(user_input)

            # 사용자가 입력한 문장을 past 리스트에 저장
            # → 채팅 UI에서 "사용자 말풍선" 표시용
            st.session_state['past'].append(user_input)

             # 챗봇이 생성한 답변을 generated 리스트에 저장
            # → 채팅 UI에서 "봇 말풍선" 표시용
            st.session_state['generated'].append(output)

# ---------------------------------------------
# 채팅 기록 화면 출력 영역
# ---------------------------------------------

# generated 리스트에 데이터가 있을 때만 출력
# (처음 실행 시 빈 화면 방지)
    if st.session_state['generated']:
        # 챗봇 출력 영역(response_container) 안에서 출력
        with response_container:
            
            # 저장된 대화 개수만큼 반복
            # past 와 generated는 항상 같은 길이를 유지
            for i in range(len(st.session_state['generated'])):
                    # -------------------------------
                    # 사용자 메시지 출력
                    # -------------------------------
                message(st.session_state["past"][i],  # i번째 사용자 입력 문장
                        is_user=True, # 사용자 메시지임을 표시
                        key=str(i) + '_user',   # Streamlit UI 충돌 방지용 고유 key
                        avatar_style = "fun-emoji",   # 사용자 아바타 스타일
                        seed = "Nala") # 아바타 고정용 시드값
                # -------------------------------
                # 챗봇 메시지 출력
                # -------------------------------
                message(st.session_state["generated"][i],# i번째 챗봇 답변
                         key=str(i),    # 고유 key
                         avatar_style = "bottts",  # 봇 스타일 아바타
                         seed = "Fluffy")# 봇 아바타 고정 시드