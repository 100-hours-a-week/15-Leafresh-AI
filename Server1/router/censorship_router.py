from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY

from pydantic import BaseModel, Field
from typing import List, Optional

from model.censor.LLM_censorship_model import CensorshipModel
from model.censor.censorship_hyperclovax_model import HyperClovaxModel

import asyncio
from concurrent.futures import ProcessPoolExecutor

router = APIRouter()
# model = CensorshipModel()
# model = HyperClovaxModel()

# 모델 경로
# GCP model_dir = "/home/ubuntu/hyperclovax_model/models--naver-hyperclovax--HyperCLOVAX-SEED-Text-Instruct-1.5B/snapshots/543a1be9d6233069842ffce73aa56a232a4f457b"
# local model_dir = "./hyperclovax_model/models--naver-hyperclovax--HyperCLOVAX-SEED-Text-Instruct-1.5B/snapshots/543a1be9d6233069842ffce73aa56a232a4f457b"
MODEL_PATH = "/home/ubuntu/hyperclovax_model/models--naver-hyperclovax--HyperCLOVAX-SEED-Text-Instruct-1.5B/snapshots/543a1be9d6233069842ffce73aa56a232a4f457b"
MODEL_INSTANCE = None

def init_model():
    global MODEL_INSTANCE
    MODEL_INSTANCE = HyperClovaxModel(MODEL_PATH)

def model_worker(challenge_name, start_date, end_date, existing_list):
    global MODEL_INSTANCE
    return MODEL_INSTANCE.validate(challenge_name, start_date, end_date, existing_list)

def warmup_task():
    return "[Warmup] Model loaded"

# CPU 코어 2개 사용 
executor = ProcessPoolExecutor(max_workers=2, initializer=init_model)

async def warmup_workers():
    loop = asyncio.get_event_loop()
    futures = [executor.submit(warmup_task) for _ in range(2)]
    await asyncio.gather(*[asyncio.wrap_future(f) for f in futures])

# 요청 데이터 모델
class ChallengeInfo(BaseModel):
    id: Optional[int] = None
    name: Optional[str] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None 

class ValidationRequest(BaseModel):
    memberId: Optional[int] = None
    challengeName: Optional[str] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    challenge: List[ChallengeInfo] = Field(default_factory=list)

# 응답 데이터 모델
class ValidationResponse(BaseModel):
    status: int
    message: str
    data: Optional[dict] = None

@router.post("/ai/challenges/group/validation", response_model=ValidationResponse)
async def validate_challenge(req: ValidationRequest):

    if req.memberId is None:
        return JSONResponse(status_code=400, content={
            "status": 400, "message": "사용자 ID는 필수 항목입니다.", "data": None
        })
    if not req.challengeName:
        return JSONResponse(status_code=400, content={
            "status": 400, "message": "챌린지 이름은 필수 항목입니다.", "data": None
        })
    if not req.startDate:
        return JSONResponse(status_code=400, content={
            "status": 400, "message": "시작 날짜는 필수 항목입니다.", "data": None
        })
    if not req.endDate:
        return JSONResponse(status_code=400, content={
            "status": 400, "message": "끝 날짜는 필수 항목입니다.", "data": None
        })

    if req.challenge is None or len(req.challenge) == 0:            # if not req.challenge: 
        return JSONResponse(
            status_code=200,
            content={
                "status": 200,
                "message": "챌린지 목록이 없으므로 생성 가능합니다.",
                "data": {
                    "result": True
                }
            }
        )

    future = executor.submit(model_worker, req.challengeName, req.startDate, req.endDate, [c.model_dump() for c in req.challenge])

    is_creatable, msg = await asyncio.wrap_future(future)

    return JSONResponse(
        status_code=200,
        content={
            "status": 200,
            "message": msg,
            "data": {
                "result": is_creatable
            }
        }
    )

# 422 예외 처리
async def validation_exception_handler(request: Request, exc: RequestValidationError):      # 사용하지 않더라도 FastAPI는 고정된 규약을 따르므로 request, exc를 유지해야함 
    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": 422,
            "message": "필수 항목이 누락되었거나 형식이 잘못되었습니다.",
            "data": None
        }
    )

# 500, 503, 이외 예외 처리
async def http_exception_handler(request: Request, exc: HTTPException):

    if exc.status_code == 500:
        message = "챌린지 유사도 분석 중 오류가 발생했습니다. 다시 시도해주세요."
    elif exc.status_code == 503:
        message = "현재 서버가 혼잡하여 요청을 처리할 수 없습니다. 잠시 후 다시 시도해주세요."
    else:
        message = exc.detail 

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": exc.status_code,
            "message": message,
            "data": None
        }
    )
