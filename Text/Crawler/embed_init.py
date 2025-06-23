# embed_init.py
# Qdrant와 SentenceTransformerEmbeddings를 사용하여 문서 임베딩 및 저장
from generate_challenge_docs import generate_challenge_docs
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, SearchParams, Filter
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv
import os
import hashlib

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
환경변수 RESET_COLLECTION이 존재하지 않으면 기본값은 "false"로 간주"
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

def get_content_hash(content):
    """문서 내용의 해시값을 생성하여 중복 체크에 사용"""
    return hashlib.md5(content.encode()).hexdigest() #MD5 해시는 내용이 조금이라도 다르면 완전히 다른 해시값 생성함

# 문서 생성 및 벡터 저장은 항상 실행
try:
    # 크롤링 및 데이터 생성
    result = generate_challenge_docs(file_path=None, mode="random")
    fixed_challenges = result["fixed_challenges"]
    crawled_challenges = result["crawled_challenges"]
    
    # 청크 분할
    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=20)
    documents = []
    seen_hashes = set()  # 중복 체크를 위한 해시값 저장
    
    # 1. 고정 데이터 처리
    for challenge in fixed_challenges:
        content_hash = get_content_hash(challenge)
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            # 고정 데이터의 메타데이터 추출
            metadata = {
                "category": (
                    "제로웨이스트" if "제로웨이스트" in challenge or "플라스틱" in challenge or "분리수거" in challenge else
                    "플로깅" if "플로깅" in challenge or "정화" in challenge or "청소" in challenge else
                    "비건" if "비건" in challenge or "채식" in challenge or "식단" in challenge else
                    "에너지절약" if "에너지" in challenge or "전기" in challenge or "냉난방" in challenge else
                    "업사이클" if "업사이클" in challenge or "재활용" in challenge or "DIY" in challenge else
                    "문화공유" if "공유" in challenge or "캠페인" in challenge or "워크숍" in challenge else
                    "디지털탄소" if "디지털" in challenge or "이메일" in challenge or "클라우드" in challenge else
                    "기타"
                ),
                "source": "기본데이터",
                "content_hash": content_hash  # 해시값을 메타데이터에 저장
            }
            documents.append(Document(page_content=challenge, metadata=metadata))
    
    # 2. 크롤링 데이터 처리
    for challenge in crawled_challenges:
        content_hash = get_content_hash(challenge["content"])
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            challenge["metadata"]["content_hash"] = content_hash  # 해시값을 메타데이터에 저장
            documents.append(Document(
                page_content=challenge["content"],
                metadata=challenge["metadata"]
            ))
    
    # 청크 분할 및 임베딩
    chunks = splitter.split_documents(documents)
    
    # 중복 제거된 청크만 저장
    unique_chunks = []
    seen_chunk_hashes = set()
    
    for chunk in chunks:
        chunk_hash = get_content_hash(chunk.page_content)
        if chunk_hash not in seen_chunk_hashes:
            seen_chunk_hashes.add(chunk_hash)
            chunk.metadata["chunk_hash"] = chunk_hash
            unique_chunks.append(chunk)
    
    # 임베딩 및 Qdrant 저장
    vectorstore.add_documents(unique_chunks)
    print("문서 임베딩 및 Qdrant 저장 완료")
    
    try:
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        vector_count = collection_info.points_count
        print(f"현재 Qdrant에 저장된 총 누적 벡터 수: {vector_count}")
    except Exception as e:
        print(f"벡터 수 조회 중 오류 발생: {str(e)}")
except Exception as e:
    print(f"문서 임베딩 중 오류 발생: {str(e)}")

# 검색 함수 수정
def search_similar_challenges(query, limit=5):
    """
    유사한 챌린지를 검색하는 함수
    Args:
        query (str): 검색 쿼리
        limit (int): 반환할 결과 수
    Returns:
        list: 검색 결과 문서 리스트
    """
    try:
        # 검색 파라미터 설정
        search_params = SearchParams(
            hnsw_ef=128,  # 검색 정확도 향상
            exact=False   # 근사 검색 사용
        )
        
        # 검색 실행
        results = vectorstore.similarity_search(
            query,
            k=limit * 2,  # 더 많은 결과를 가져와서 필터링
            search_params=search_params
        )
        
        # 중복 제거 및 카테고리 다양성 확보
        seen_hashes = set()
        unique_results = []
        category_count = {}
        
        for doc in results:
            content_hash = doc.metadata.get("chunk_hash", get_content_hash(doc.page_content))
            category = doc.metadata.get("category", "기타")
            
            # 중복 제거 및 카테고리 분포 확인
            if content_hash not in seen_hashes:
                if category not in category_count:
                    category_count[category] = 0
                
                # 각 카테고리당 최대 2개까지만 포함
                if category_count[category] < 2:
                    seen_hashes.add(content_hash)
                    unique_results.append(doc)
                    category_count[category] += 1
                
                if len(unique_results) >= limit:
                    break
        
        return unique_results[:limit]
        
    except Exception as e:
        print(f"검색 중 오류 발생: {str(e)}")
        return []
