from dotenv import load_dotenv
import json

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# LangChain OpenAI
from langchain_openai import ChatOpenAI


load_dotenv()



def basic_chain_setting():
    llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)

    rag_prompt = PromptTemplate.from_template(
        """
너는 생활 법률 전문 챗봇이다.

반드시 제공된 법률 문서(context_text)에 포함된 내용만을 근거로 답변해야 한다.

❗ 허용 규칙
- 문서에 있는 내용을 벗어나지 않는 범위에서
  문장의 순서를 바꾸거나 표현을 쉽게 바꾸는 것은 허용된다.
- 의미를 유지한 요약 및 쉬운 말 변환은 허용된다.

❗ 금지 규칙
- 문서에 없는 새로운 정보 추가 금지
- 문서에 없는 조항 번호 생성 금지
- 일반 상식으로 내용 보충 금지

---

사용자 질문:
{question}

관련 법률 문서:
{context_text}

---

### 답변 작성 규칙

1. 문서에서 질문과 관련된 조항을 모두 찾는다.

2. 관련 조항을 다음 형식으로 그대로 제시한다.

[관련 법령 및 조항]
- 법령명 제○조(조항 제목)
(문서 원문 내용)

3. 위 조항 내용만을 근거로 의미를 유지한 상태에서
비전공자가 이해하기 쉽게 풀어서 설명한다.

[법률 설명]
- 전문 용어는 쉬운 말로 바꿔 설명한다.
- 문서 의미를 벗어나지 않는다.
- 예시는 문서 의미 범위 안에서만 사용한다.
"""
    )

    rag_chain = rag_prompt | llm | StrOutputParser()
    return rag_chain


def classify_chain_setting():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

    prompt = PromptTemplate.from_template(
        """
        너는 질문 분류기다.

아래 질문을 다음 3가지 중 하나로 분류하라.

--------------------------------------------------

[law]

아래 법률과 직접 관련된 질문 또는 상황 설명이면 law로 분류한다.

① 근로기준법 관련
- 임금, 월급, 체불, 해고, 퇴직, 근로계약, 근무시간, 연장근로, 야근, 휴게시간,
  산업재해, 직장 내 괴롭힘, 폭행, 폭언, 성희롱, 징계, 해고예고, 퇴직금

② 주택임대차보호법 관련
- 전세, 월세, 보증금, 계약갱신청구권, 집주인, 임대인, 임차인, 보증금 반환,
  전입신고, 확정일자, 전세계약, 임대차계약

③ 도로교통법 관련
- 교통사고, 음주운전, 신호위반, 과속, 벌점, 면허정지, 면허취소, 단속,
  횡단보도, 주차위반

④ 개인정보 보호법 관련
- 개인정보, 주민등록번호, 유출, 동의, 수집, 저장, 파기, CCTV, 마케팅 문자,
  광고 문자, 동의 철회

⑤ 전자상거래 소비자보호법 관련
- 온라인 쇼핑, 환불, 반품, 교환, 결제취소, 배송지연, 사기, 중고거래,
  쿠팡, 네이버쇼핑, 쇼핑몰, 판매자, 구매자

또는 위 법들과 관련된 **분쟁 상황 설명**도 law로 분류한다.

--------------------------------------------------

[simple]

일반 대화, 감정 표현, 인사, 잡담, 조언 요청(법률과 무관)일 경우 simple.

--------------------------------------------------

[impossible]

법률관련 상황이아니거나 법률관련질문이 아니면 impossible.

--------------------------------------------------

⚠️ 규칙
- 반드시 law / simple / impossible 중 하나만 출력
- 설명하지 말고 단어만 출력

질문: {question}
"""
    )

    return prompt | llm | StrOutputParser()


def simple_chain_setting():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4, streaming=True)
    simple_prompt = PromptTemplate.from_template(
        """
    너는 사용자의 일반적인 일상 대화를 담당하는 챗봇이다.

    허용 질문:
    - 인사, 잡담, 감정 표현
    - 날씨, 시간, 일정, 일반 정보 질문
    - 가벼운 조언 요청


    최근 대화:
    {context}

    질문:
    {question}

    답변:
"""
   
    )

    simple_chain = simple_prompt | llm | StrOutputParser()

    return simple_chain

def impossible_chain_setting():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4, streaming=True)
    imp_prompt = PromptTemplate.from_template(
        """
    너는 답변할 수 없는 질문에 대응하는 챗봇이다.

    다음 기준에 해당하지 않는 질문에만 답변 불가하다고 답변한다.
    - 생활법률(근로기준법, 주택임대차보호법, 도로교통법 ,개인정보 보호법 ,전자상거래 소비자보호법) 질문
    - 법률(근로기준법 주택임대차보호법 도로교통법 개인정보 보호법 전자상거래 소비자보호법)과 관련된 상황 설명 질문
    - 일상적인 대화 질문
    
    [예시답변]:
    "해당 질문은 현재 답변할 수 없습니다.
    법률 관련 질문이나 일상 질문만 답변 가능합니다."

    최근 대화 내용:
    {context}

    사용자 질문:
    {question}


    """
        )

    imp_chain = imp_prompt | llm | StrOutputParser()

    return imp_chain
