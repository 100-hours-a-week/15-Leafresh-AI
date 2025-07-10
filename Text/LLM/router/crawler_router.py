from fastapi import APIRouter
from Text.Crawler.generate_challenge_docs import generate_challenge_docs

router = APIRouter()

@router.post("/ai/crawler/run")
async def run_crawler():
    try:
        generate_challenge_docs()
        return {"status": "ok", "message": "크롤러 실행 완료"}
    except Exception as e:
        return {"status": "error", "message": str(e)} 

"""
curl -X POST http://localhost:8000/ai/crawler/run #터미널에서 입력시 크롤링 실핼
"""