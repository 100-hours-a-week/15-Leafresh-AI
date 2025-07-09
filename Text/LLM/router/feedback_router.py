from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY, HTTP_202_ACCEPTED, HTTP_500_INTERNAL_SERVER_ERROR
from pydantic import BaseModel, field_validator
from typing import List, Dict, Any
from datetime import datetime
import json
import logging
from dotenv import load_dotenv
import os
from google.cloud import pubsub_v1

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

router = APIRouter()

# GCP Pub/Sub 설정
project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "leafresh-dev2")
topic_id = os.getenv("PUBSUB_TOPIC_FEEDBACK_DEV")  # leafresh-feedback-topic

topic_path = f"projects/{project_id}/topics/{topic_id}"
publisher = pubsub_v1.PublisherClient()

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
    try:
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

        # 요청 데이터 준비
        request_data = request.model_dump()
        request_data["timestamp"] = datetime.now().isoformat()
        request_data["requestId"] = f"req_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # GCP Pub/Sub 토픽으로 메시지 발행
        message_json = json.dumps(request_data, ensure_ascii=False, default=str)
        message_bytes = message_json.encode("utf-8")
        
        future = publisher.publish(topic_path, data=message_bytes)
        message_id = future.result()
        
        logger.info(f"[PUBLISH] 피드백 요청 발행 완료 (message ID: {message_id})")
        logger.info(f"[PUBLISH] Request Data: {json.dumps(request_data, ensure_ascii=False, default=str)}")
        
        # 유효한 요청인 경우, 즉시 202 Accepted 응답 반환
        return JSONResponse(
            status_code=202,
            content={
                "status": 202,
                "message": "피드백 요청이 정상적으로 접수되었습니다. 결과는 추후 Pub/Sub으로 전송됩니다.",
                "data": {
                    "requestId": request_data["requestId"],
                    "messageId": message_id
                }
            }
        )
        
    except Exception as e:
        logger.error(f"[ERROR] 피드백 요청 발행 실패: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": 500,
                "message": "피드백 요청 발행 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
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