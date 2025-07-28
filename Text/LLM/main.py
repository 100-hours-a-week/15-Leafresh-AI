from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError, HTTPException
from fastapi import Request
from fastapi.responses import JSONResponse
import sys
import os
import uvicorn
import threading
import logging

load_dotenv()
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from router.error_router import validation_exception_handler
from router.error_router import http_exception_handler

from Text.LLM.router.chatbot_router import router as chatbot_router

from Text.LLM.router.feedback_router import router as feedback_router
from Text.LLM.router.feedback_router import feedback_exception_handler
from Text.LLM.router.feedback_router import feedback_http_exception_handler

from Text.LLM.router.crawler_router import router as crawler_router

from fastapi.middleware.cors import CORSMiddleware

from prometheus_client import start_http_server

# 모니터링 미들웨어 import
from router.monitoring_router import metrics_middleware

# SQS 워커 import
from model.feedback.sqs_feedback_worker import run_worker

if __name__ == "__main__":
    
    from Text.Crawler.generate_challenge_docs import generate_challenge_docs
    
    logger.info("Leafresh AI 서버 시작 중...")
    
    # 크롤러를 백그라운드에서 실행
    logger.info("크롤러 백그라운드 실행 시작")
    # threading.Thread(target=generate_challenge_docs, daemon=True).start()
    
    # SQS 피드백 워커를 백그라운드에서 실행
    logger.info("SQS 피드백 워커 백그라운드 실행 시작")
    threading.Thread(target=run_worker, daemon=True).start()
    
    # 9104 포트에서 exporter 실행
    logger.info("Prometheus 메트릭 서버 시작 (포트: 9104)")
    start_http_server(9104)
    
    logger.info("FastAPI 서버 시작 (포트: 8000)")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

# app 초기화
app = FastAPI()

# 메트릭 미들웨어
# 모든 요청의 시작과 끝을 측정해야 하기에 먼저 설정
app.middleware("http")(metrics_middleware)

# CORS 미들웨어
origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# router 등록
app.include_router(chatbot_router)
app.include_router(feedback_router)
app.include_router(crawler_router)

# 글로벌 예외 핸들러 등록 (라우팅 기반)
@app.exception_handler(RequestValidationError)
async def global_validation_handler(request: Request, exc: RequestValidationError):
    if request.url.path.startswith("/ai/feedback"):
        return await feedback_exception_handler(request, exc)
    elif request.url.path.startswith("/ai/challenges/group/validation"):
        return await validation_exception_handler(request, exc)
    else:
        return JSONResponse(
            status_code=422,
            content={
                "status": 422,
                "message": "유효하지 않은 요청입니다.",
                "data": None
            }
        )

@app.exception_handler(HTTPException)
async def global_http_exception_handler(request: Request, exc: HTTPException):
    if request.url.path.startswith("/ai/feedback"):
        return await feedback_http_exception_handler(request, exc)
    elif request.url.path.startswith("/ai/challenges/group/validation"):
        return await http_exception_handler(request, exc)
    else:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "status": exc.status_code,
                "message": exc.detail,
                "data": None
            }
        )

