# feedback_router.py
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY, HTTP_202_ACCEPTED, HTTP_500_INTERNAL_SERVER_ERROR
from pydantic import BaseModel
from typing import List, Dict, Any
import uuid
import time
from model.feedback.publisher_be_to_ai_aws import add_feedback_task

router = APIRouter()

# Define Pydantic models for the input data based on API spec
class Submission(BaseModel):
    submittedAt: str  # Changed to str to accept ISO format string
    isSuccess: bool

class PersonalChallenge(BaseModel):
    # Added Optional for id and title based on API spec examples
    id: int | None = None
    title: str | None = None
    isSuccess: bool

class GroupChallenge(BaseModel):
    # Added Optional for id, title, startDate, endDate based on API spec examples
    id: int | None = None
    title: str | None = None
    startDate: str | None = None  # Changed to str to accept ISO format string
    endDate: str | None = None    # Changed to str to accept ISO format string
    submissions: List[Submission] = [] 

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
     # 요청을 SQS로 발행 (비동기 처리)
    try:
        req_id = str(uuid.uuid4())
        payload: Dict[str, Any] = {
            "memberId": request.memberId,
            "personalChallenges": [pc.model_dump() for pc in request.personalChallenges],
            "groupChallenges": [gc.model_dump() for gc in request.groupChallenges],
            "timestamp": int(time.time()),
            "requestId": req_id
        }
        add_feedback_task(payload) # SQS에 비동기적으로 발행
        print(f"[REQUEST] 피드백 요청 발행 (RequestId: {req_id})")
    except Exception as e:
        # 발행 실패 시 500 반환
        return JSONResponse(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": 500,
                "message": f"피드백 요청 발행 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
                "data": None
            }
        )      
    
    # API 명세에 따른 202 응답
    return JSONResponse(
        status_code=HTTP_202_ACCEPTED,
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
