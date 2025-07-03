from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError, HTTPException
from fastapi import Request
from fastapi.responses import JSONResponse
import sys
import os
import uvicorn

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from router.error_router import validation_exception_handler
from router.error_router import http_exception_handler

from Text.LLM.router.chatbot_router import router as chatbot_router

from Text.LLM.router.feedback_router import router as feedback_router
from Text.LLM.router.feedback_router import feedback_exception_handler
from Text.LLM.router.feedback_router import feedback_http_exception_handler

from fastapi.middleware.cors import CORSMiddleware

from prometheus_client import start_http_server

if __name__ == "__main__":
    # 9104 포트에서 exporter 실행
    start_http_server(9104)
    uvicorn.run("main:app", host="0.0.0.0", port=8000)

load_dotenv()

# app 초기화
app = FastAPI()

# CORS 미들웨어 추가
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

