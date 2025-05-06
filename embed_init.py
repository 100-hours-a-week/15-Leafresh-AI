# embed_init.py
# Qdrant와 SentenceTransformerEmbeddings를 사용하여 문서 임베딩 및 저장
from generate_challenge_docs import generate_challenge_docs
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()

# 환경변수 로드
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")

# Qdrant 클라이언트
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# 현재 존재하는 컬렉션 목록 조회
try:
    existing_collections = qdrant_client.get_collections().collections
    existing_names = [coll.name for coll in existing_collections]
    print(f"현재 Qdrant에 존재하는 컬렉션 목록: {existing_names}")
except Exception as e:
    print(f"컬렉션 목록 조회 중 오류 발생: {str(e)}")

# 콜렉션 없으면 새로 생성
try:
    collections = qdrant_client.get_collections().collections
    collection_names = [coll.name for coll in collections]

    if not qdrant_client.collection_exists(COLLECTION_NAME):
        print(f" '{COLLECTION_NAME}' 컬렉션 생성")
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
    else:
        print(f" '{COLLECTION_NAME}' 컬렉션이 이미 존재합니다.")
except Exception as e:
    print(f"컬렉션 생성 중 오류 발생: {str(e)}")

# 임베딩 모델
embedding_fn = SentenceTransformerEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# Qdrant vectorstore 객체
vectorstore = Qdrant(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embeddings=embedding_fn,
)

RESET_COLLECTION = os.getenv("RESET_COLLECTION", "false").lower() == "true"
"""
환경변수 RESET_COLLECTION이 존재하지 않으면 기본값은 "false"로 간주”
컬렉션을 리셋할지 여부 (환경변수 또는 코드로 지정)
환경변수로 설정된 경우 우선 적용
예: export RESET_COLLECTION=true
기존 컬렉션을 완전히 삭제 후 새로 생성
"""

if RESET_COLLECTION:
    try:
        print(f"기존 컬렉션 '{COLLECTION_NAME}' 삭제 중")
        qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        print(f"컬렉션 초기화 완료: '{COLLECTION_NAME}'")
    except Exception as e:
        print(f"컬렉션 초기화 중 오류 발생: {str(e)}")

# 문서 생성 및 벡터 저장은 항상 실행
try:
    generate_challenge_docs(file_path="challenge_docs.txt", mode="random")
    documents = TextLoader("challenge_docs.txt").load()
    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=20)
    chunks = splitter.split_documents(documents)
    vectorstore.add_documents(chunks)
    print("문서 임베딩 및 Qdrant 저장 완료")
    try:
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        vector_count = collection_info.points_count
        print(f"현재 Qdrant에 저장된 총 누적 벡터 수: {vector_count}")
    except Exception as e:
        print(f"벡터 수 조회 중 오류 발생: {str(e)}")
except Exception as e:
    print(f"문서 임베딩 중 오류 발생: {str(e)}")
