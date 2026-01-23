# -------------------------------
# Streamlit 웹앱 라이브러리 불러오기
# → 파이썬 코드만으로 웹 화면(UI)을 만들 수 있음
# -------------------------------
import streamlit as st

# -------------------------------
# PDF 파일을 읽기 위한 라이브러리
# → PDF 열기, 페이지 접근, 텍스트 추출 가능
# -------------------------------
from PyPDF2 import PdfReader

# -------------------------------
# OpenAI 임베딩 모델 불러오기
# → 텍스트를 숫자 벡터로 변환 (검색/유사도 비교용)
# -------------------------------
from langchain_openai import OpenAIEmbeddings


# -------------------------------
# OpenAI ChatGPT 모델 불러오기
# → 질문 보내고 답변 받는 역할
# -------------------------------
from langchain_classic.chat_models import ChatOpenAI

# -------------------------------
# 문서 기반 질문응답 체인 불러오기
# -------------------------------

# ConversationalRetrievalChain
# → 이전 대화 기억 + 문서 검색 + GPT 답변 생성
# RetrievalQA
# → 문서 검색 + GPT 답변 생성 (대화 기억 없음)
from langchain_classic.chains import ConversationalRetrievalChain, RetrievalQA

# -------------------------------
# 대화 메모리 관리 클래스
# → 최근 대화를 저장해서 챗봇이 문맥을 이해하도록 도움
# -------------------------------
from langchain_classic.memory import ConversationBufferWindowMemory

# -------------------------------
# 벡터 데이터베이스 (FAISS)
# → 임베딩 벡터를 저장하고 빠르게 검색하기 위해 사용
# -------------------------------
from langchain_classic.vectorstores import FAISS

# -------------------------------
# PDF 파일을 LangChain Document 형태로 불러오는 로더
# → PDF + 메타데이터 함께 관리 가능
# -------------------------------
from langchain_classic.document_loaders import PyPDFLoader

# -------------------------------
# 긴 텍스트를 작은 조각으로 나누는 도구
# → GPT 입력 제한 문제 해결 + 검색 정확도 향상
# -----------------------------
from langchain_classic.text_splitter  import RecursiveCharacterTextSplitter

# -------------------------------
# 운영체제(OS) 환경 변수 접근용 라이브러리
# -------------------------------
import os

# -------------------------------
# .env 파일에 저장된 비밀키(API KEY) 불러오는 라이브러리
# -------------------------------
from dotenv import load_dotenv

# -------------------------------
# .env 파일을 메모리로 로드
# -------------------------------
load_dotenv()

# -------------------------------
# 환경변수에서 OpenAI API 키 가져오기
# .env 파일 예시:
# OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
# -------------------------------
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")


# =====================================================
# PDF 파일에서 텍스트만 추출하는 함수
# =====================================================
def get_pdf_text(pdf_docs):
     # PDF에서 추출한 텍스트를 저장할 빈 문자열 생성
    text=""
    for pdf in pdf_docs:
        # PDF 파일 열기
        pdf_reader=PdfReader(pdf)
        # 현재 PDF의 모든 페이지 반복
        for page in pdf_reader.pages:
             # 각 페이지의 텍스트 추출 후 기존 text에 이어붙이기
            text +=page.extract_text()
     # 모든 PDF의 텍스트를 하나로 합쳐서 반환
    return text

# =====================================================
# 긴 텍스트를 작은 조각(chunk)으로 나누는 함수
# =====================================================
def get_text_chunks(text):
     # -----------------------------------------
    # RecursiveCharacterTextSplitter 생성
    # → LangChain에서 제공하는 텍스트 분할 도구
    # -----------------------------------------
    text_splitter=RecursiveCharacterTextSplitter(
        # 텍스트를 나눌 기준 문자
        # "\n" = 줄바꿈(엔터) 기준으로 우선 분리
        separators="\\n",

        # 하나의 chunk(조각) 최대 길이
        # 단위: 글자 수
        # → GPT 입력 제한과 검색 효율을 고려한 크기
        chunk_size=1000,

        # chunk 간 겹치는 글자 수
        # → 문맥이 끊기지 않도록 앞부분을 일부 중복 저장
        chunk_overlap=200,

        # chunk 간 겹치는 글자 수
        # → 문맥이 끊기지 않도록 앞부분을 일부 중복 저장
        length_function=len
    )
    # -----------------------------------------
    # 실제로 텍스트를 나누는 부분
    # -----------------------------------------

    # 긴 텍스트를 여러 개 chunk 리스트로 변환
    chunks=text_splitter.split_text(text)

    # -----------------------------------------
    # 분할된 텍스트 조각 리스트 반환
    # -----------------------------------------
    return chunks



# =====================================================
# 텍스트 chunk들을 벡터 DB(FAISS)로 변환하는 함수
# =====================================================
def get_vectorstore(text_chunks):
    # -----------------------------------------
    # OpenAI 임베딩 모델 생성
    # -----------------------------------------

    # 텍스트를 숫자 벡터로 변환하는 모델
    # → 문장 의미를 수치화해서 유사도 검색 가능하게 만듦
    embeddings=OpenAIEmbeddings(model='text-embedding-ada-002')# OpenAI에서 제공하는 대표적인 임베딩 모델
    # -----------------------------------------
    # FAISS 벡터 데이터베이스 생성
    # -----------------------------------------

    # from_texts 함수는 내부적으로 다음 작업을 자동 수행:
    # 1. text_chunks 하나씩 읽기
    # 2. embeddings 모델로 벡터 변환
    # 3. 변환된 벡터들을 FAISS DB에 저장
    vectorstore=FAISS.from_texts(texts=text_chunks, # 분할된 텍스트 리스트
                                 embedding=embeddings)# 사용할 임베딩 모델
    
    # -----------------------------------------
    # 완성된 벡터 DB 반환
    # -----------------------------------------
    return vectorstore


# =====================================================
# 벡터 DB(FAISS)를 이용한 대화형 문서 QA 체인을 생성하는 함수
# =====================================================
def get_conversation_chain(vectorstore):

    # -------------------------------------------------
    # 1. 대화 메모리 객체 생성
    # -------------------------------------------------

    # ConversationBufferWindowMemory는
    # 사용자의 이전 질문과 AI의 답변을 저장해서
    # 다음 질문에서도 문맥을 유지하도록 도와줌
    memory=ConversationBufferWindowMemory(
                                        memory_key='chat_history', # LangChain 내부에서 대화 기록을 저장할 변수 이름
                                         # 대화 기록을 Message 객체 형태로 반환
                                        # → ConversationalRetrievalChain과 호환을 위해 True 설정
                                          return_message=True)
    # -------------------------------------------------
    # 2. ConversationalRetrievalChain 생성
    # -------------------------------------------------

    # GPT 모델 + 벡터 검색기 + 메모리를 하나로 묶는 핵심 코드
    conversation_chain=ConversationalRetrievalChain.from_llm(
         # ---------------------------
        # GPT 모델 설정
        # --------------------------
        llm=ChatOpenAI(
            # temperature=0 → 랜덤성 제거
            # → 문서 기반 QA에서는 정확한 답변이 중요하기 때문
            temperature=0,
             # 사용할 OpenAI GPT 모델 이름
            # gpt-4.1-mini → 빠르고 비용 효율적인 최신 경량 모델
            model_name='gpt-4.1-mini'),
        # ---------------------------
        # 검색기(retriever) 설정
        # ---------------------------

        # FAISS 벡터 DB를 검색기 형태로 변환
        # → 질문이 들어오면 벡터 유사도 기반으로
        #    가장 관련 있는 문서 chunk들을 찾아줌
        retriever=vectorstore.as_retriever(),

         # ---------------------------
        # 대화 기록 처리 방식 설정
        # ---------------------------

        # lambda h: h 는
        # "기존 대화 기록을 그대로 사용하겠다"는 의미
        # LangChain 내부 포맷 문제를 방지하기 위한 설정
        get_chat_history=lambda h:h,

        # ---------------------------
        # 메모리 연결
        # ---------------------------

        # 위에서 만든 대화 메모리 객체를 체인에 연결
        memory=memory
    )
    # -------------------------------------------------
    # 3. 완성된 체인 반환
    # -------------------------------------------------

    # 이제 이 객체 하나로:
    # - 질문 전달
    # - 문서 검색
    # - GPT 답변 생성
    # - 대화 기억
    # 을 모두 처리 가능
    return conversation_chain

# 파일 업로드 위젯 생성
# 화면에 "파일을 업로드해주세요 ~" 라는 문구와 함께 파일 선택 버튼이 생김
# accept_multiple_files=True → 여러 개 파일 업로드 가능
user_uploads=st.file_uploader("파일을 업로드해주세요 ~",accept_multiple_files=True)

# 사용자가 파일을 업로드 했을 경우에만 실행
# (아무 파일도 없으면 None 이므로 아래 코드는 실행 안됨)
if user_uploads is not None:
    # "Upload" 버튼 생성
    # 버튼을 클릭했을 때만 내부 코드가 실행됨
    if st.button("Upload"):
        # 처리 중임을 사용자에게 보여주는 로딩 스피너
        # "처리중.." 이라는 문구와 함께 로딩 애니메이션 표시
        with st.spinner("처리중.."):

            # 업로드된 PDF 파일들에서 텍스트를 추출
            # user_uploads → PDF 파일 객체 리스트
            # raw_text → PDF에서 뽑아낸 전체 텍스트 문자열
            raw_text=get_pdf_text(user_uploads)

            # 추출한 텍스트를 작은 덩어리(chunk)로 분할
            # 이유: LLM은 긴 문장을 한번에 처리하기 어려움
            # text_chunks → ["문서 조각1", "문서 조각2", ...]
            text_chunks=get_text_chunks(raw_text)

            # 텍스트 조각들을 벡터로 변환하여 벡터 데이터베이스 생성
            # AI가 질문에 맞는 문서를 빠르게 검색하기 위한 저장소
            # vectorstore → FAISS / Chroma 와 같은 벡터 DB 객체
            vectorstore=get_vectorstore(text_chunks)

            # 벡터스토어를 기반으로 질문-응답 체인 생성
            # LangChain QA 체인 생성
            # session_state에 저장해서 페이지 전체에서 재사용 가능
            st.session_state.conversation=get_conversation_chain(vectorstore)


# 채팅 입력창 생성 (ChatGPT 스타일 입력칸)
# 사용자가 질문을 입력하고 엔터를 치면 실행됨
# := 는 입력값을 user_query 변수에 저장하면서 동시에 조건 검사
if user_query := st.chat_input("질문을 입력해주세요~"):
    # 문서를 업로드해서 conversation 객체가 생성되었는지 확인
    # 업로드 없이 질문하면 에러 나기 때문에 체크
    if 'conversation' in st.session_state:

         # LangChain 대화 체인 실행
        # question → 사용자 질문
        # chat_history → 이전 대화 기록 (없으면 빈 리스트)
        result=st.session_state.conversation({
            "question":user_query,
            "chat_history":st.session_state.get("chat_history",[])
        })

        # 결과 딕셔너리에서 AI 답변만 추출
        # result 구조 예:
        # {
        #   "answer": "답변 내용",
        #   "source_documents": [...]
        # }
        response=result['answer']
    else:
         # 문서를 업로드 하지 않고 질문한 경우
        response="먼저 문서를 업로드해주세요~"

     # AI 말풍선 출력 영역 생성
    # assistant → 챗봇(인공지능) 말풍선
    with st.chat_message("assistant"):
        # 답변을 화면에 출력
        st.write(response)
