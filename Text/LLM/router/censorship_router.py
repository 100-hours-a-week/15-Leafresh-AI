from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY

from pydantic import BaseModel, Field
from typing import List, Optional

from ..model.censorship.LLM_censorship_model import CensorshipModel

router = APIRouter()
model = CensorshipModel()

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

    if req.challenge is None or len(req.challenge) == 0:
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

    is_creatable, msg = model.validate(
        req.challengeName, req.startDate, req.endDate, req.challenge
    )
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
