import os
from dotenv import load_dotenv
from pymongo import MongoClient
import certifi  # 공식 루트 인증서 체인 명시적 지정 

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME")

# MongoDB 클라이언트 생성
client = MongoClient(MONGO_URI, tls=True, tlsAllowInvalidCertificates=True, tlsCAFile=certifi.where())  # GCP에 올릴 때는 'tlsAllowInvalidCertificates=True'부분 삭제
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# 데이터 삽입
def insert_prompt(challenge_id: int, challenge_name: str, prompt_text: str):
    doc = {
        "challenge_id": challenge_id,
        "challenge_name": challenge_name,
        "prompt": prompt_text
    }
    result = collection.insert_one(doc)

    return result.inserted_id

# 데이터 조회
def get_prompt_by_id(challenge_id: int):
    result = collection.find_one({"challenge_id": challenge_id})

    return result["prompt"] if result else None

# 중복 확인 함수
def prompt_exists(challenge_id: int):
    return collection.find_one({"challenge_id": challenge_id}) is not None

