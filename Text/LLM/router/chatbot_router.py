# chatbot_router.py
from ..model.chatbot.LLM_chatbot_base_info_model import base_prompt, get_llm_response
from ..model.chatbot.LLM_chatbot_free_text_model import qa_chain, retriever, process_chat, clear_conversation, conversation_states
from ..model.chatbot.chatbot_constants import label_mapping, ENV_KEYWORDS, BAD_WORDS
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from fastapi import HTTPException
from pydantic import BaseModel
from typing import Optional
import json
import re

router = APIRouter()

class CategoryRequest(BaseModel):
    sessionId: Optional[str] = None
    location: Optional[str] = None
    workType: Optional[str] = None
    category: Optional[str] = None
class FreeTextRequest(BaseModel):
    sessionId: Optional[str] = None
    message: Optional[str] = None

# 비-RAG 방식 챌린지 추천
@router.post("/ai/chatbot/recommendation/base-info")
def select_category(req: CategoryRequest):
    missing_fields = []
    # 필수 필드 검사
    if not req.location:
        missing_fields.append("location")
    if not req.workType:
        missing_fields.append("workType")
    if not req.category:
        missing_fields.append("category")
    if missing_fields:
        return JSONResponse(
            status_code=400,
            content={
                "status": 400,
                "message": f"{missing_fields}은 필수입니다.",
                "data": None
            }
        )

    # LLM 호출을 위한 prompt 구성
    prompt = base_prompt.format(
        location=req.location,
        workType=req.workType,
        category=req.category
    )

    try:
        parsed = get_llm_response(prompt)
        if req.sessionId:
            process_chat(req.sessionId, f"카테고리: {req.category}, 위치: {req.location}, 직업: {req.workType}", base_info_category=req.category)
        if req.category not in label_mapping:
            return JSONResponse(
                status_code=400,
                content={
                    "status": 400,
                    "message": "유효하지 않은 선택 항목이 포함되어 있습니다.",
                    "data": {
                        "invalidFields": ["category"]
                    }
                }
            )
        eng_label, kor_label = label_mapping[req.category]

        return JSONResponse(
            status_code=200,
            content={
                "status": 200,
                "message": "사용자 기본 정보 키워드 선택을 기반으로 챌린지를 추천합니다.",
                "data": {
                    "recommend": parsed.get("recommend", ""),
                    "challenges": [
                        {
                            "title": c.get("title", ""),
                            "description": c.get("description", ""),
                            "category": eng_label,
                            "label": kor_label
                        }
                        for c in parsed.get("challenges", [])
                    ] if isinstance(parsed.get("challenges"), list) else []
                }
            }
        )
    except HTTPException as http_err:
        raise http_err # 내부 HTTPException을 먼저 처리
    except Exception as e:
        return JSONResponse(
            status_code=502,
            content={
                "status": 502,
                "message": "AI 서버로부터 추천 결과를 받아오는 데 실패했습니다.",
                "data": None
            }
        )

# LangChain 기반 RAG 추천
@router.post("/ai/chatbot/recommendation/free-text")
def freetext_rag(req: FreeTextRequest):
    missing_fields = []
    if not req.message or not req.message.strip():
        missing_fields.append("message")

    if missing_fields:
        return JSONResponse(
            status_code=400,
            content={
                "status": 400,
                "message": f"{missing_fields}은 필수입니다.",
                "data": None
            }
        )
    if len(req.message.strip()) < 5:
        return JSONResponse(
            status_code=422,
            content={
                "status": 422,
                "message": "message는 문자열이어야 하며, 최소 5자 이상의 문자열이어야 합니다.",
                "data": None
            }
        )
        
    message_lower = req.message.lower()
    
    # 카테고리 관련 요청 체크
    category_keywords = ["원래", "처음", "이전", "원래대로", "기존", "카테고리"]
    is_category_request = any(keyword in message_lower for keyword in category_keywords)
    
    # 환경 관련 요청이 아니고, 카테고리 요청도 아닌 경우에만 기본 응답
    if not is_category_request and (not any(k in req.message for k in ENV_KEYWORDS) or any(b in message_lower for b in BAD_WORDS)):
        return JSONResponse(
            status_code=200,
            content={
                "status": 200,
                "message": "사용자에게 자유 메세지를 기반으로 챌린지를 추천합니다.",
                "data": {
                    "recommend": "저는 친환경 챌린지를 추천해드리는 Leafresh 챗봇이에요! 환경 관련 질문을 해주시면 더 잘 도와드릴 수 있어요.",
                    "challenges": None
                }
            }
        )
    
    try:
        # 현재 세션의 카테고리 확인
        current_category = None
        if req.sessionId in conversation_states:
            state = conversation_states[req.sessionId]
            if "category" in state:
                current_category = state["category"]
                print(f"Current category from state: {current_category}")

        # 대화 기록을 포함한 응답 생성
        response_text = process_chat(req.sessionId, req.message, base_info_category=current_category)
        
        try:
            # response_text는 이미 JSON 문자열이므로 바로 파싱
            parsed = json.loads(response_text)
            
            # 현재 세션의 최신 카테고리 가져오기
            if req.sessionId in conversation_states:
                current_category = conversation_states[req.sessionId]["category"]
                print(f"Using latest category from state: {current_category}")
                
                if current_category in label_mapping:
                    eng_label, kor_label = label_mapping[current_category]
                    print(f"Using category labels - eng: {eng_label}, kor: {kor_label}")
                    
                    # 챌린지 데이터에 현재 카테고리 정보 업데이트
                    if "challenges" in parsed:
                        for challenge in parsed["challenges"]:
                            challenge["category"] = eng_label
                            challenge["label"] = kor_label
            
            return JSONResponse(
                status_code=200,
                content={
                    "status": 200,
                    "message": "사용자 자유 메시지를 기반으로 챌린지를 추천합니다.",
                    "data": {
                        "recommend": parsed.get("recommend", ""),
                        "challenges": parsed.get("challenges", [])
                    }
                }
            )
        except json.JSONDecodeError:
            return JSONResponse(
                status_code=500,
                content={
                    "status": 500,
                    "message": "서버 내부 오류로 인해 챌린지 추천에 실패했습니다.",
                    "data": None
                }
            )
            
    except Exception as e:
        return JSONResponse(
            status_code=502,
            content={
                "status": 502,
                "message": f"AI 서버로부터 추천 결과를 받아오는 데 실패했습니다.", # AI 서버 오류
                "data": None
            }
        )