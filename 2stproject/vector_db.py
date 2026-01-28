import os
import json
import re
from langchain_classic.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


DATA_DIR = "./data"                 # 네 data 폴더
CHROMA_DIR = "./chroma_law_db"      # 벡터 DB 저장 위치

documents = []

for filename in os.listdir(DATA_DIR):
    if not filename.endswith(".json"):
        continue

    file_path = os.path.join(DATA_DIR, filename)
    print(file_path)
    # --------------------
    # 1. 법 / 시행령 구분 & 법 이름 정규화
    # --------------------
    raw_name = filename.split("(")[0].strip()
    law_type = "시행령" if "시행령" in raw_name else "법"
    law_name = raw_name.replace(" 시행령", "")
    print(raw_name,law_type,law_name)

    # --------------------
    # 2. JSON 로드
    # --------------------
    with open(file_path, "r", encoding="utf-8") as f:
        law_json = json.load(f)

    # --------------------
    # 3. 조 → 항 단위 분해
    # --------------------
    for article in law_json:
        title = article["title"]                  # 제1조(목적)
        
        # Regex로 '제N조' 또는 '제N조의M' 추출
        match = re.match(r"^(제\d+조(?:의\d+)?)", title)
        if match:
            article_num = match.group(1)
        else:
            article_num = title.split("(")[0].strip()

        article_keywords = article.get("keywords", [])
        # 리스트 첫 번째 요소를 제목으로 사용 (없으면 빈 문자열)
        article_title = article_keywords[0] if article_keywords else ""

        for idx, para in enumerate(article["contents"], start=1):
            paragraph = f"제{idx}항"

            # --------------------
            # text (임베딩 대상)
            # --------------------
            text = f"[{law_name} {article_num} {paragraph}]\n"
            text += para["text"]

            if para.get("items"):
                for item in para["items"]:
                    text += f"\n- {item}"

            # --------------------
            # metadata (필터용)
            # --------------------
            metadata = {
                "law_name": law_name,
                "law_type": law_type,
                "article": article_num,
                "article_title": article_title,
                "paragraph": paragraph,
                "source_file": filename
            }

            documents.append(
                Document(
                    page_content=text,
                    metadata=metadata
                )
            )

embedding = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY")
)
vectordb = Chroma.from_documents(
    documents=documents,
    embedding=embedding,
    persist_directory=CHROMA_DIR
)

vectordb.persist()

print(f"✅ Chroma DB 저장 완료: {len(documents)}개 문서")
