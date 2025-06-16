from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError, HTTPException
from fastapi import Request
from fastapi.responses import JSONResponse
import threading
from contextlib import asynccontextmanager

from model.verify.worker import run_worker
from router.verify_router import router as verify_router

from router.censorship_router import router as censorship_router
from router.censorship_router import validation_exception_handler, http_exception_handler

from router.chatbot_router import router as chatbot_router

from router.feedback_router import router as feedback_router
from router.feedback_router import feedback_exception_handler
from router.feedback_router import feedback_http_exception_handler

from router.health_router import router as health_router

load_dotenv()

# worker를 main 실행할 때 지속적으로 실행되도록 변경 
# pubsub_v1이 동기로 실행되므로 async를 붙이지 않음 
@asynccontextmanager
async def lifespan(app: FastAPI):               # app 인자를 받는 형태가 아니면 에러가 발생하므로 삭제 불가능 
    threading.Thread(target=run_worker, daemon=True).start()
    yield

# app 초기화
app = FastAPI(lifespan=lifespan)

# router 등록
app.include_router(verify_router)
app.include_router(censorship_router)
app.include_router(chatbot_router)
app.include_router(feedback_router)
app.include_router(health_router)

# censorship model exceptions (422, 500, 503 etc.)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(HTTPException, http_exception_handler)

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