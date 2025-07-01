from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY, HTTP_202_ACCEPTED, HTTP_500_INTERNAL_SERVER_ERROR
from pydantic import BaseModel, field_validator
from typing import List, Dict, Any
from datetime import datetime
# 실제 파일 구조에 맞게 import 경로 조정
from ..model.feedback.LLM_feedback_model import FeedbackModel
import httpx # httpx 라이브러리 임포트
import os # 환경 변수 로드를 위해 os 임포트
from rq import Queue
from redis import Redis
from Text.LLM.model.feedback.tasks import generate_feedback_task

router = APIRouter()

# Redis 연결 및 큐 생성 (전역에서 한 번만)
redis_conn = Redis(host='localhost', port=6379, db=0)
feedback_queue = Queue('feedback', connection=redis_conn)

# API 명세에 따른 입력 데이터를 위한 Pydantic 모델 정의
class Submission(BaseModel):
    isSuccess: bool
    submittedAt: datetime

    @field_validator('submittedAt', mode='before')
    @classmethod
    def parse_submitted_at(cls, v):
        if isinstance(v, list):
            # [년, 월, 일, 시, 분, 초, 마이크로초]
            if len(v) == 7:
                # 마이크로초가 999999를 초과하는 경우 조정
                microsecond = min(v[6], 999999)
                return datetime(v[0], v[1], v[2], v[3], v[4], v[5], microsecond)
            return datetime(*v)
        return v

class PersonalChallenge(BaseModel):
    id: int | None = None
    title: str | None = None
    isSuccess: bool

class GroupChallenge(BaseModel):
    id: int | None = None
    title: str | None = None
    startDate: datetime
    endDate: datetime
    submissions: List[Submission] = []

    @field_validator('startDate', 'endDate', mode='before')
    @classmethod
    def parse_date(cls, v):
        if isinstance(v, list):
            # [년, 월, 일, 시, 분, 초]
            return datetime(*v)
        return v

class FeedbackRequest(BaseModel):
    memberId: int 
    personalChallenges: List[PersonalChallenge]  # Optional 제거
    groupChallenges: List[GroupChallenge]

@router.post("/ai/feedback")
async def create_feedback(request: FeedbackRequest):
    # API 명세상 챌린지 데이터가 하나라도 누락된 경우 400 응답
    if not request.personalChallenges and not request.groupChallenges:
        return JSONResponse(
            status_code=400,
            content={
                "status": 400,
                "message": "요청 값이 유효하지 않습니다. 챌린지 데이터가 모두 포함되어야 합니다.",
                "data": None
            }
        )

    # 유효한 요청인 경우, 즉시 202 Accepted 응답 반환
    job = feedback_queue.enqueue('Text.LLM.model.feedback.tasks.generate_feedback_task', request.model_dump())
    return JSONResponse(
        status_code=202,
        content={
            "status": 202,
            "message": "피드백 요청이 정상적으로 접수되었습니다. 결과는 추후 콜백으로 전송됩니다.",
            "data": None
        }
    )

# 예외 핸들러 함수들
async def feedback_exception_handler(request: Request, exc: RequestValidationError):
    print(f"Validation Error: {exc.errors()}")  # 에러 상세 내용 출력
    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": 422,
            "message": "필수 항목이 누락되었거나 형식이 잘못되었습니다.",
            "data": None
        }
    )

async def feedback_http_exception_handler(request: Request, exc: HTTPException):
    if exc.status_code == HTTP_500_INTERNAL_SERVER_ERROR:
        message = "서버 오류로 피드백 생성을 완료하지 못했습니다. 잠시 후 다시 시도해주세요."
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