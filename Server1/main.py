from dotenv import load_dotenv
from fastapi import FastAPI

import threading
from contextlib import asynccontextmanager
import multiprocessing

# from model.verify.worker import run_worker
from model.verify.worker_aws import run_worker
from router.verify_router import router as verify_router
from router.health_router import router as health_router
# from router.llava_router import router as llava_router
from router.censorship_router import router as censorship_router

from fastapi.exceptions import RequestValidationError, HTTPException
from router.censorship_router import validation_exception_handler, http_exception_handler
from router.censorship_router import warmup_workers

from prometheus_client import start_http_server
from router.monitoring_router import metrics_middleware

load_dotenv()

# 모니터링 서버 실행 함수
def run_metrics_server():
    start_http_server(9101)

multiprocessing.set_start_method("spawn", force=True)

# worker를 main 실행할 때 지속적으로 실행되도록 변경 
# pubsub_v1이 동기로 실행되므로 async를 붙이지 않음 
@asynccontextmanager
async def lifespan(app: FastAPI):               # app 인자를 받는 형태가 아니면 에러가 발생하므로 삭제 불가능 
    print("[DEBUG] Lifespan 시작됨")

    # censorship 모델 워커 로딩 (여기서 fork 발생 가능)
    print("[INFO] FastAPI 서버 시작됨: censorship 모델 워커 초기화 시작")
    await warmup_workers()
    print("[INFO] censorship 모델 워커 2개 초기화 완료")

    # gRPC 구독 워커 실행 (fork 이후에 실행되어야 안전)
    threading.Thread(target=run_worker, daemon=True).start()
    threading.Thread(target=run_metrics_server, daemon=True).start()    # 메트릭 서버를 별도 스레드에서 실행 
    print("[DEBUG] run_worker() 스레드 시작됨")

    yield
    print("[INFO] FastAPI 서버 종료")


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

# 모니터링
app.middleware("http")(metrics_middleware)