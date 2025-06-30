from dotenv import load_dotenv
from fastapi import FastAPI

import threading
from contextlib import asynccontextmanager

from model.verify.worker import run_worker
from router.verify_router import router as verify_router
from router.health_router import router as health_router
# from router.llava_router import router as llava_router

from fastapi.exceptions import RequestValidationError, HTTPException
from router.censorship_router import router as censorship_router
from router.censorship_router import validation_exception_handler, http_exception_handler

load_dotenv()

# worker를 main 실행할 때 지속적으로 실행되도록 변경 
# pubsub_v1이 동기로 실행되므로 async를 붙이지 않음 
@asynccontextmanager
async def lifespan(app: FastAPI):               # app 인자를 받는 형태가 아니면 에러가 발생하므로 삭제 불가능 
    print("[DEBUG] Lifespan 시작됨")
    threading.Thread(target=run_worker, daemon=True).start()
    yield

# app 초기화
app = FastAPI(lifespan=lifespan)

# router 등록
app.include_router(censorship_router)
app.include_router(verify_router)
app.include_router(health_router)
# app.include_router(llava_router)

# censorship model exceptions (422, 500, 503 etc.)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(HTTPException, http_exception_handler)