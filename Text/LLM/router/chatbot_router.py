# chatbot_router.py
from ..model.chatbot.LLM_chatbot_base_info_model import base_prompt, get_llm_response as get_base_info_llm_response, base_parser
from ..model.chatbot.LLM_chatbot_free_text_model import process_chat, clear_conversation, conversation_states, custom_prompt, get_llm_response as get_free_text_llm_response
from ..model.chatbot.chatbot_constants import label_mapping, ENV_KEYWORDS, BAD_WORDS
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from typing import Optional, Dict, Any, Generator
import json
import re
from fastapi.responses import StreamingResponse
from typing import Generator, AsyncGenerator
import asyncio
from urllib.parse import unquote  # URL 디코딩을 위한 import 추가

router = APIRouter()

# 비-RAG 방식 챌린지 추천 (SSE)
@router.get("/ai/chatbot/recommendation/base-info")
async def select_category(
    sessionId: Optional[str] = Query(None),
    location: Optional[str] = Query(None),
    workType: Optional[str] = Query(None),
    category: Optional[str] = Query(None)
):
    """
    사용자의 기본 정보를 기반으로 친환경 챌린지를 추천하는 SSE 엔드포인트
    """

    
    # URL 디코딩
    if location:
        location = unquote(location)
    if workType:
        workType = unquote(workType)
    if category:
        category = unquote(category)
    
    # 입력값 검증
    if not location:
        print("location 누락")
        raise HTTPException(
            status_code=400,
            detail={
                "status": 400,
                "message": "location은 필수입니다.",
                "data": None
            }
        )
    if not workType:
        print("workType 누락")
        raise HTTPException(
            status_code=400,
            detail={
                "status": 400,
                "message": "workType은 필수입니다.",
                "data": None
            }
        )
    if not category:
        print("category 누락")
        raise HTTPException(
            status_code=400,
            detail={
                "status": 400,
                "message": "category는 필수입니다.",
                "data": None
            }
        )

    # 유효한 카테고리 검증
    if category not in label_mapping:
        print(f"유효하지 않은 카테고리: {category}")
        print(f"유효한 카테고리 목록: {list(label_mapping.keys())}")
        raise HTTPException(
            status_code=400,
            detail={
                "status": 400,
                "message": "유효하지 않은 선택 항목이 포함되어 있습니다.",
                "data": {
                    "invalidFields": ["category"]
                }
            }
        )

    # LLM 호출을 위한 prompt 구성
    prompt = base_prompt.format(
        location=location,
        workType=workType,
        category=category
    )

    # SSE 응답 생성
    async def event_generator():
        # 세션 정보만 저장 (RAG 사용하지 않음)
        if sessionId and sessionId not in conversation_states:
            conversation_states[sessionId] = {
                "messages": [],
                "category": category,
                "base_category": category
            }

        try:
            # 전체 응답 텍스트를 누적하기 위한 변수 (base_info에서는 LLM_chatbot_base_info_model에서 파싱하므로 제거)
            # full_response = ""
            eng_label = label_mapping[category]
            
            for data_payload in get_base_info_llm_response(prompt, category=category):
                event_type = data_payload.get("event")
                data_from_llm_model = data_payload.get("data")
                
                if event_type == "challenge":
                    # LLM_chatbot_base_info_model에서 이미 json.dumps 처리된 문자열을 받음
                    yield {
                        "event": "challenge",
                        "data": data_from_llm_model,
                        "id": None,
                        "retry": None
                    }
                    
                elif event_type == "close":
                    try:
                        # LLM 모델에서 이미 파싱된 데이터를 받음 (json.dumps 처리된 문자열)
                        # 여기서는 문자열을 다시 파싱해서 딕셔너리로 만듦
                        parsed_data = json.loads(data_from_llm_model) if isinstance(data_from_llm_model, str) else data_from_llm_model
                        
                        # 파싱된 데이터가 유효한지 확인
                        if not isinstance(parsed_data, dict):
                            raise ValueError("파싱된 데이터가 딕셔너리가 아닙니다.")
                            
                        # 'data' 키 안에 실제 데이터가 있는지 확인
                        if "data" not in parsed_data or not isinstance(parsed_data["data"], dict):
                            raise ValueError("파싱된 데이터에 유효한 'data' 키가 없습니다.")
                            
                        final_data = parsed_data["data"]

                        if "challenges" not in final_data:
                            raise ValueError("파싱된 데이터에 'challenges' 키가 없습니다.")
                            
                        # LLM_chatbot_base_info_model에서 이미 카테고리/라벨이 추가되므로 여기서는 추가하지 않음
                        # for challenge in final_data["challenges"]:
                        #     challenge["category"] = eng_label
                        #     challenge["label"] = kor_label
                        
                        yield {
                            "event": "close",
                            "data": json.dumps({
                                "status": 200,
                                "message": "모든 챌린지 추천 완료",
                                "data": final_data
                            }, ensure_ascii=False),
                            "id": None,
                            "retry": None
                        }
                    except Exception as e:
                        raise HTTPException(
                            status_code=500,
                            detail={
                                "status": 500,
                                "message": f"파싱 실패: {str(e)}",
                                "data": None
                            }
                        )
                    
                elif event_type == "error":
                    yield {
                        "event": "error",
                        "data": data_from_llm_model,
                        "id": None,
                        "retry": None
                    }
                
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={
                    "status": 500,
                    "message": f"서버 내부 오류로 추천에 실패했습니다: {str(e)}",
                    "data": None
                }
            )

    return EventSourceResponse(event_generator())

# LangChain 기반 RAG 추천
@router.get("/ai/chatbot/recommendation/free-text")
async def freetext_rag(
    sessionId: Optional[str] = Query(None),
    message: Optional[str] = Query(None)
):
    """
    자유 채팅 입력을 기반으로 친환경 챌린지를 추천하는 SSE 엔드포인트
    """
    # 입력값 검증
    if not message or not message.strip():
        raise HTTPException(
            status_code=400,
            detail={
                "status": 400,
                "message": "message는 필수입니다.",
                "data": None
            }
        )

    if len(message.strip()) < 5:
        raise HTTPException(
            status_code=422,
            detail={
                "status": 422,
                "message": "message는 문자열이어야 하며, 최소 5자 이상의 문자열이어야 합니다.",
                "data": None
            }
        )

    message_lower = message.lower()
    
    # 카테고리 관련 요청 체크
    category_keywords = ["원래", "처음", "이전", "원래대로", "기존", "카테고리"]
    is_category_request = any(keyword in message_lower for keyword in category_keywords)
    
    # 환경 관련 요청이 아니고, 카테고리 요청도 아닌 경우에만 기본 응답 (fallback 로직)
    is_env_related = any(k in message for k in ENV_KEYWORDS)
    contains_bad_words = any(b in message_lower for b in BAD_WORDS)

    async def event_generator():
        # 세션 초기화 (free-text는 대화 기록이 중요)
        if sessionId not in conversation_states:
            conversation_states[sessionId] = {
                "messages": [],
                "category": "제로웨이스트", # 기본값
                "base_category": "제로웨이스트"
            }
        
        # fallback 조건 검사
        if not is_category_request and (not is_env_related or contains_bad_words):
            yield {
                "event": "fallback",
                "data": json.dumps({
                    "status": 200,
                    "message": "저는 친환경 챌린지를 추천해드리는 Leafresh 챗봇이에요! 환경 관련 질문을 해주시면 더 잘 도와드릴 수 있어요.",
                    "data": None
                }, ensure_ascii=False)
            }
            return # fallback 메시지 전송 후 종료

        # LLM 호출을 위한 프롬프트 구성
        current_category = conversation_states[sessionId].get("category", "제로웨이스트")
        messages_history = "\n".join(conversation_states[sessionId]["messages"])
        
        prompt = custom_prompt.format(
            context="",  # RAG는 현재 비활성화
            query=message,
            messages=messages_history,
            category=current_category
        )

        # LLM 응답 스트리밍
        try:
            full_response_text = ""
            for data_payload in get_free_text_llm_response(prompt):
                event_type = data_payload.get("event")
                data_from_llm_model = data_payload.get("data")

                if event_type == "token":
                    # LLM_chatbot_free_text_model에서 이미 딕셔너리로 받음
                    yield {
                        "event": "token",
                        "data": json.dumps(data_from_llm_model, ensure_ascii=False)
                    }
                    if "data" in data_from_llm_model and "token" in data_from_llm_model["data"]:
                        full_response_text += data_from_llm_model["data"]["token"]
                        
                elif event_type == "complete":
                    # LLM_chatbot_free_text_model에서 이미 딕셔너리로 받음
                    # 대화 기록 업데이트 (챌린지 추천 결과 전체를 저장)
                    if sessionId in conversation_states:
                        conversation_states[sessionId]["messages"].append(f"AI: {full_response_text}")

                    yield {
                        "event": "complete",
                        "data": json.dumps(data_from_llm_model, ensure_ascii=False)
                    }
                    yield {
                        "event": "close",
                        "data": json.dumps({
                            "status": 200,
                            "message": "모든 응답 완료",
                            "data": data_from_llm_model["data"] # complete 이벤트의 data를 그대로 사용
                        }, ensure_ascii=False)
                    }
                    
                elif event_type == "error":
                    yield {
                        "event": "error",
                        "data": json.dumps(data_from_llm_model, ensure_ascii=False)
                    }
                
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={
                    "status": 500,
                    "message": f"서버 내부 오류로 추천에 실패했습니다: {str(e)}",
                    "data": None
                }
            )

    return EventSourceResponse(event_generator())

# # 세션 초기화 엔드포인트 추가 (필요 시)
# @router.post("/ai/chatbot/clear-conversation")
# async def clear_chat_history(sessionId: str = Query(...)):
#     clear_conversation(sessionId)
#     return JSONResponse(
#         status_code=200,
#         content={
#             "status": 200,
#             "message": f"Session {sessionId} 대화 기록이 초기화되었습니다.",
#             "data": None
#         }
#     )