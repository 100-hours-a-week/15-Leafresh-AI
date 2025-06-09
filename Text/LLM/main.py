from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError, HTTPException
from fastapi import Request
from fastapi.responses import JSONResponse
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from Text.LLM.router.censorship_router import router as censorship_router
from Text.LLM.router.censorship_router import validation_exception_handler
from Text.LLM.router.censorship_router import http_exception_handler

from Text.LLM.router.chatbot_router import router as chatbot_router

from Text.LLM.router.feedback_router import router as feedback_router
from Text.LLM.router.feedback_router import feedback_exception_handler
from Text.LLM.router.feedback_router import feedback_http_exception_handler

load_dotenv()

# app 초기화
app = FastAPI()

# router 등록
app.include_router(censorship_router)
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

