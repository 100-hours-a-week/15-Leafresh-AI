# embed_init.py
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from dotenv import load_dotenv
import os

load_dotenv()

# ✅ 환경변수 로드
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "my-new-collection"  # ✅ 여기 명시

# Qdrant 클라이언트
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# ✅ 콜렉션 없으면 새로 생성
try:
    collections = qdrant_client.get_collections().collections
    collection_names = [coll.name for coll in collections]

    if not qdrant_client.collection_exists(COLLECTION_NAME):
        print(f"📦 '{COLLECTION_NAME}' 컬렉션 생성")
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
    else:
        print(f"✅ '{COLLECTION_NAME}' 컬렉션이 이미 존재합니다.")
except Exception as e:
    print(f"❌ 컬렉션 생성 중 오류 발생: {str(e)}")

# 임베딩 모델
embedding_fn = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Qdrant vectorstore 객체
vectorstore = Qdrant(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embeddings=embedding_fn,
)


retriever = vectorstore.as_retriever()

# ✅ 문서 임베딩 및 저장
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

try:
    documents = TextLoader("challenge_docs.txt").load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    vectorstore.add_documents(chunks)
    print("✅ 문서 임베딩 및 Qdrant 저장 완료")
    try:
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        vector_count = collection_info.points_count
        print(f"📊 현재 Qdrant에 저장된 벡터 수: {vector_count}")
    except Exception as e:
        print(f"❌ 벡터 수 조회 중 오류 발생: {str(e)}")
except Exception as e:
    print(f"❌ 문서 임베딩 중 오류 발생: {str(e)}")

__all__ = ["embedding_fn", "vectorstore", "retriever", "qdrant_client", "COLLECTION_NAME"]