from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from model.verify.LLM_verify_model import ImageVerifyModel
from model.verify.pubsub_helper import add_task

router = APIRouter()
verifier = ImageVerifyModel()

# 요청 데이터 모델
class ImageVerificationRequest(BaseModel):
    verificationId: int
    type: str
    imageUrl: str           
    memberId: int
    challengeId: int
    date: str
    challengeName: str

# 응답 데이터 모델
class ImageVerificationResponse(BaseModel):
    status: int
    message: str
    data: Optional[dict] = None

@router.post("/ai/image/verification", response_model=ImageVerificationResponse, status_code=202)
async def verify_image(req: ImageVerificationRequest):
    try:
        data = req.model_dump()         # req.dict() -> req.model_dump()
        data["date"] = str(req.date)
        add_task(data)

        return JSONResponse(
            status_code=202,
            content={
                "status": 202,
                "message": "이미지 인증 요청이 정상적으로 접수되었습니다. 결과는 추후 콜백으로 전송됩니다.",
                "data": None
            }
        )
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": 500,
                "message": f"이미지 인증 중 오류 발생: {e}",
                "data": None
            }
        )

class CallbackResult(BaseModel):
    type: str
    memberId: int
    challengeId: int
    date: str
    result: bool

@router.post("/api/verifications/{verificationId}/result")
async def receive_result(verificationId: int, data: CallbackResult):
    print(f"콜백 수신 완료: {data}")
    return JSONResponse(
        status_code=200,
        content={
            "status": "received",
            "verificationId": verificationId
        }
    )
