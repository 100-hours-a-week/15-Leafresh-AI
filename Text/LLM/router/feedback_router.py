from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY, HTTP_202_ACCEPTED, HTTP_500_INTERNAL_SERVER_ERROR
from pydantic import BaseModel
from typing import List, Dict, Any

# Adjust the import based on your actual file structure
from ..model.feedback.LLM_feedback_model import FeedbackModel

import httpx # httpx 라이브러리 임포트
import os # 환경 변수 로드를 위해 os 임포트

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
async def create_feedback(request: FeedbackRequest, background_tasks: BackgroundTasks):
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

    # 실제 피드백 생성 및 처리 로직을 수행할 함수
    async def run_feedback_generation(data: Dict[str, Any]):
        CALLBACK_URL = os.getenv("CALLBACK_URL_FEEDBACK")
        if not CALLBACK_URL:
            print("CALLBACK_URL_FEEDBACK 환경 변수가 설정되지 않았습니다. 피드백 결과를 전송할 수 없습니다.")
            return

        callback_url = f"{CALLBACK_URL}/api/members/feedback/result"

        try:
            feedback_model = FeedbackModel()
            feedback_result = await feedback_model.generate_feedback(data)

            if feedback_result and feedback_result.get("status") == 200:
                callback_payload = {
                    "memberId": data.get("memberId"),
                    "content": feedback_result.get("data", {}).get("feedback", "")
                }
                print(f"BE 서비스로 피드백 결과 전송 시도: {callback_url} with payload {callback_payload}")
                async with httpx.AsyncClient() as client:
                    callback_response = await client.post(callback_url, json=callback_payload)
                    callback_response.raise_for_status()
                    print(f"피드백 결과 BE 전송 성공: 상태 코드 {callback_response.status_code}")
            elif feedback_result:
                print(f"피드백 모델 오류 발생. 결과를 BE에 전송하지 않습니다. 응답: {feedback_result}")
            else:
                print("피드백 모델 응답이 유효하지 않습니다.")

        except httpx.HTTPStatusError as http_err:
            print(f"BE 서비스 콜백 중 HTTP 오류 발생: {http_err}")
        except httpx.RequestError as req_err:
            print(f"BE 서비스 콜백 중 요청 오류 발생: {req_err}")
        except Exception as e:
            print(f"백그라운드 피드백 생성/전송 중 예상치 못한 오류 발생: {e}")

    # 유효한 요청인 경우, 즉시 202 Accepted 응답 반환
    background_tasks.add_task(run_feedback_generation, request.model_dump())

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