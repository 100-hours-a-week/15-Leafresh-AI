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

router = APIRouter()

def format_sse_response_for_client(event: str, data: Dict[str, Any]) -> str:
    """클라이언트에 보낼 최종 SSE 응답 형식으로 변환 (data는 Python 딕셔너리)"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

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
    # 입력값 검증
    if not location:
        raise HTTPException(
            status_code=400,
            detail={
                "status": 400,
                "message": "location은 필수입니다.",
                "data": None
            }
        )
    if not workType:
        raise HTTPException(
            status_code=400,
            detail={
                "status": 400,
                "message": "workType은 필수입니다.",
                "data": None
            }
        )
    if not category:
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
            # 동기 제너레이터를 직접 사용
            full_response = ""
            for data_payload in get_base_info_llm_response(prompt):
                event_type = data_payload.get("event_type", "message")
                
                if event_type == "challenge":
                    challenge = data_payload.get("data", {})
                    eng_label, kor_label = label_mapping[category]
                    
                    yield format_sse_response_for_client("challenge", {
                        "status": 200,
                        "message": f"{challenge.get('index', '')} 번째 챌린지 추천",
                        "data": {
                            "challenges": {
                                "title": challenge.get("title", ""),
                                "description": challenge.get("description", ""),
                                "category": eng_label,
                                "label": kor_label
                            }
                        }
                    })
                    
                elif event_type == "close":
                    try:
                        parsed_data = base_parser.parse(full_response.strip())
                        yield format_sse_response_for_client("close", {
                            "status": 200,
                            "message": "모든 챌린지 추천 완료",
                            "data": parsed_data
                        })
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
                    raise HTTPException(
                        status_code=500,
                        detail={
                            "status": 500,
                            "message": data_payload.get("message", "알 수 없는 오류가 발생했습니다."),
                            "data": None
                        }
                    )
                
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
            yield format_sse_response_for_client("fallback", {
                "status": 200,
                "message": "저는 친환경 챌린지를 추천해드리는 Leafresh 챗봇이에요! 환경 관련 질문을 해주시면 더 잘 도와드릴 수 있어요.",
                "data": None
            })
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

        full_response_text = ""
        try:
            # 동기 제너레이터를 직접 사용
            for data_payload in get_free_text_llm_response(prompt):
                event_type = data_payload.get("event_type", "message")
                
                if event_type == "token":
                    token = data_payload.get("data", {}).get("token", "")
                    full_response_text += token
                    # 각 토큰을 즉시 yield
                    yield format_sse_response_for_client("token", {
                        "status": 200,
                        "message": "토큰 생성",
                        "data": {
                            "token": token
                        }
                    })
                    
                elif event_type == "error":
                    raise HTTPException(
                        status_code=500,
                        detail={
                            "status": 500,
                            "message": data_payload.get("message", "알 수 없는 오류가 발생했습니다."),
                            "data": None
                        }
                    )
                
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={
                    "status": 500,
                    "message": f"LLM 응답 스트리밍 중 오류 발생: {str(e)}",
                    "data": None
                }
            )

        # 전체 응답 텍스트를 JSON으로 파싱하고 챌린지 이벤트 생성
        try:
            json_match = re.search(r'```json\s*(\{.*\})\s*```', full_response_text, re.DOTALL)
            if json_match:
                json_string = json_match.group(1)
            else:
                json_string = full_response_text
                
            parsed_response = json.loads(json_string)

            if "challenges" in parsed_response and isinstance(parsed_response["challenges"], list):
                # 대화 기록 업데이트 (성공적인 LLM 응답만 기록)
                conversation_states[sessionId]["messages"].append(f"User: {message}")
                conversation_states[sessionId]["messages"].append(f"Assistant: {json.dumps(parsed_response, ensure_ascii=False)}")
                if len(conversation_states[sessionId]["messages"]) > 10:
                    conversation_states[sessionId]["messages"] = conversation_states[sessionId]["messages"][-10:]

                # 각 챌린지를 개별 'challenge' 이벤트로 전송
                for idx, challenge in enumerate(parsed_response["challenges"]):
                    message_text = ""
                    if idx == 0: message_text = "첫 번째 챌린지 추천"
                    elif idx == 1: message_text = "두 번째 챌린지 추천"
                    elif idx == 2: message_text = "세 번째 챌린지 추천"
                    else: message_text = f"{idx+1} 번째 챌린지 추천"

                    # 카테고리 정보는 세션 상태에서 가져오거나 기본값 사용
                    challenge_category = conversation_states[sessionId].get("category", "제로웨이스트")
                    eng_label, kor_label = label_mapping.get(challenge_category, ("UNKNOWN", "알 수 없음"))

                    yield format_sse_response_for_client("challenge", {
                        "status": 200,
                        "message": message_text,
                        "data": {
                            "challenges": {
                                "title": challenge.get("title", "제목 없음"),
                                "description": challenge.get("description", "설명 없음"),
                                "category": eng_label,
                                "label": kor_label
                            }
                        }
                    })
                # 모든 챌린지 전송 후 'close' 이벤트 전송
                yield format_sse_response_for_client("close", {
                    "status": 200,
                    "message": "모든 챌린지 추천 완료",
                    "data": parsed_response
                })
            else:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "status": 500,
                        "message": "LLM 응답에서 챌린지 정보를 파싱할 수 없습니다. 응답 형식 오류.",
                        "data": None
                    }
                )

        except json.JSONDecodeError as jde:
            raise HTTPException(
                status_code=500,
                detail={
                    "status": 500,
                    "message": f"LLM 응답 JSON 파싱 오류: {str(jde)}. 원본 응답: {full_response_text[:200]}...",
                    "data": None
                }
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={
                    "status": 500,
                    "message": f"응답 처리 중 예상치 못한 오류 발생: {str(e)}",
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