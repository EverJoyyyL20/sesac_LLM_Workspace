import os
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


def load_chroma_similarity_retriever(
    chroma_path: str,
    embed_model_name: str = "BAAI/bge-m3",
    device: str = "cpu",
    k: int = 3
):
    """
    저장된 Chroma 벡터 DB를 로드하고
    similarity 기준 Retriever 반환
    """

    # =========================
    # 1️⃣ Embedding 모델 로드
    # =========================

    embedding_model = HuggingFaceEmbeddings(
        model_name=embed_model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

    # =========================
    # 2️⃣ Chroma DB 로드
    # =========================

    vectorstore = Chroma(
        persist_directory=chroma_path,
        embedding_function=embedding_model
    )

    print("✅ Chroma DB 로드 완료")

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    print("✅ Retriever 로드 완료")

    return retriever
