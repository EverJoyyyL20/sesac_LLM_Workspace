# ==============================
# 운영체제(OS) 기능을 사용하기 위한 모듈
# (환경변수 읽기, 경로 처리 등에 사용)
import os
# .env 파일에 저장된 환경 변수를 불러오기 위한 라이브러리
from dotenv import load_dotenv
# .env 파일 내용을 프로그램 환경 변수로 로드
load_dotenv()

# 환경 변수에 저장된 OPENAI_API_KEY 값을 가져옴
# .env 파일 안에:
# OPENAI_API_KEY=sk-xxxxxxx
# 이렇게 저장되어 있음
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")


# ==============================
# PDF 파일을 읽기 위한 라이브러리
from PyPDF2 import PdfReader
# 웹 화면(UI)을 만들기 위한 Streamlit 라이브러리
import streamlit as st
# 긴 텍스트를 작은 조각(chunk)으로 나누는 도구
from langchain_classic.text_splitter import CharacterTextSplitter
# OpenAI 임베딩 모델 사용 (텍스트 → 숫자 벡터 변환)
from langchain_openai import OpenAIEmbeddings
# 벡터를 저장하고 검색하기 위한 FAISS 데이터베이스
from langchain_classic import FAISS
# 질문-답변 체인 생성용 (현재 코드에서는 아직 사용 안함)
from langchain_classic.chains.question_answering import load_qa_chain
# OpenAI GPT 모델 사용 (챗봇용)
from langchain_classic.chat_models import ChatOpenAI
# OpenAI API 사용량(토큰) 추적용 콜백
from langchain_classic.callbacks import get_openai_callback


# ==============================
# 텍스트를 받아서 벡터 DB(FAISS)로 변환하는 함수
def process_text(text):
      # --------------------------
    # 텍스트 분할기 생성
    # 긴 문서를 작은 덩어리로 나누기 위한 설정
    text_splitter=CharacterTextSplitter(
         # 줄바꿈(\n)을 기준으로 분리
        separator="\n",
         # 하나의 텍스트 조각(chunk) 최대 길이
        # 1000 글자씩 잘라줌
        chunk_size=1000,
         # 이전 조각과 겹칠 글자 수
        # 문맥 유지를 위해 200글자 겹침
        chunk_overlap=200,
        # 텍스트 길이를 계산하는 함수
        # 파이썬 기본 len() 함수 사용
        length_function=len
    )
    
    # --------------------------
    # 실제 텍스트를 여러 개의 chunk로 분할
    # 결과는 리스트 형태
    # 예: ["텍스트1", "텍스트2", "텍스트3"...]
    chunks=text_splitter.split_text(text)
    # --------------------------
    # OpenAI 임베딩 모델 생성
    # 텍스트를 숫자 벡터로 변환하는 모델
    embeddings=OpenAIEmbeddings(model='text-embedding-ada-002')

    # --------------------------
    # FAISS 벡터 데이터베이스 생성
    # 각 chunk를 임베딩으로 변환해서 DB에 저장
    documents=FAISS.from_texts(chunks, # 나눈 텍스트 조각들
                               embeddings)# 사용할 임베딩 모델
    
      # --------------------------
    # 생성된 벡터 DB 반환
    # 나중에 질문 검색할 때 사용
    return documents


# ============================
# 프로그램의 메인 함수
# Streamlit 앱이 실행될 때 가장 먼저 호출됨
def main():
     # ----------------------------
    # 웹 페이지 상단 제목 표시
    # 브라우저 화면에 큰 제목으로 출력됨
    st.title("📄pdf 요약하기")
    # ----------------------------
    # 화면 구분선 추가 (UI 정리용)
    st.divider()

    # ----------------------------
    # PDF 파일 업로드 버튼 생성
    # type='pdf' → PDF 파일만 업로드 가능
    pdf=st.file_uploader("pdf파일을 업로드해주세요",type='pdf')

    # ----------------------------
    # 사용자가 PDF 파일을 업로드했을 경우에만 실행
    # 업로드 전에는 pdf 값이 None
    if pdf is not None:
         # ------------------------
        # 업로드된 PDF 파일을 읽기 위한 객체 생성
        pdf_reader=PdfReader(pdf)
        # ------------------------
        # PDF 전체 텍스트를 저장할 빈 문자열 생성
        text=""
         # ------------------------
        # PDF의 모든 페이지를 하나씩 반복
        for page in pdf_reader.pages:
            # --------------------
            # 각 페이지에서 텍스트만 추출해서
            # text 변수에 계속 누적
            text += page.extract_text()

         # ------------------------
        # PDF에서 추출한 전체 텍스트를
        # 벡터 데이터베이스(FAISS)로 변환
        # (process_text 함수는 텍스트 분할 + 임베딩 + DB 생성)
        documents=process_text(text)
        # ------------------------
        # GPT에게 전달할 요약 요청 질문 생성
        query="업로드된 PDF파일의 내용을 약 3-5 문장으로 요약해주세요"

        # ------------------------
        # query가 존재할 경우 실행
        # (문자열이므로 항상 True지만 안전장치)
        if query:
             # --------------------
            # 벡터 DB에서 질문과 가장 관련있는 문서 검색
            # PDF에서 중요한 부분만 추출
            docs=documents.similarity_search(query)
             # --------------------
            # OpenAI GPT 모델 생성
            # temperature=0.1 → 창의성 낮추고 정확한 요약 유도
            llm=ChatOpenAI(model='gpt-4.1-mini',temperature=0.1)
                        # --------------------
            # 질문-답변 체인 생성
            # chain_type='stuff'
            # → 검색된 문서를 모두 합쳐 GPT에게 전달
            chain=load_qa_chain(llm,chain_type='stuff')

             # --------------------
            # OpenAI API 사용량(토큰, 비용) 추적 시작
            with get_openai_callback() as cost:
                 # ----------------
                # GPT 실행
                # input_documents → 검색된 PDF 내용
                # question → 요약 요청 질문
                response=chain.run(input_documents=docs,question=query)
                # ----------------
                # 터미널(콘솔)에 토큰 사용량 및 비용 출력
                print(cost)
            # --------------------
            # 화면에 결과 제목 출력
            st.subheader("-- 요약 결과---")
             # --------------------
            # GPT가 생성한 요약 결과 화면에 표시
            st.write(response)

# ============================
# 이 파일이 직접 실행될 때만 main() 실행
if __name__=='__main__':
    main()        