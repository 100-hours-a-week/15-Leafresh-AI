# chatbot_router.py
from ..model.chatbot.LLM_chatbot_base_info_model import base_prompt, get_llm_response as get_base_info_llm_response, base_parser
from ..model.chatbot.LLM_chatbot_free_text_model import process_chat, clear_conversation, conversation_states, custom_prompt, get_llm_response as get_free_text_llm_response, retriever
from ..model.chatbot.chatbot_constants import label_mapping, ENV_KEYWORDS, BAD_WORDS, category_keywords
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from typing import Optional, Dict, Any, Generator
import json
import re
import logging
from fastapi.responses import StreamingResponse
from typing import Generator, AsyncGenerator
import asyncio
from urllib.parse import unquote  # URL 디코딩을 위한 import 추가

# 로깅 설정
logger = logging.getLogger(__name__)

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

    # LLM 호출을 위한 prompt 구성 (단순화된 프롬프트)
    prompt = base_prompt.format(
        location=location,
        workType=workType,
        category=category
    )

    # SSE 응답 생성
    def event_generator():
        try:
            # 세션 정보만 저장 (RAG 사용하지 않음)
            if sessionId and sessionId not in conversation_states:
                conversation_states[sessionId] = {
                    "messages": [],
                    "category": category,
                    "base_category": category
                }

            # 전체 응답 텍스트를 누적하기 위한 변수
            eng_label = label_mapping[category]

            for data_payload in get_base_info_llm_response(prompt, category=category):
                try:
                    event_type = data_payload.get("event")
                    data_from_llm_model = data_payload.get("data")

                    if not event_type or not data_from_llm_model:
                        continue  # 유효하지 않은 이벤트는 건너뛰기

                    if event_type == "challenge":
                        yield {
                            "event": "challenge",
                            "data": data_from_llm_model
                        }

                    elif event_type == "close":
                        try:
                            parsed_json_data = json.loads(data_from_llm_model) if isinstance(data_from_llm_model, str) else data_from_llm_model

                            if not isinstance(parsed_json_data, dict):
                                raise ValueError("파싱된 데이터가 딕셔너리가 아닙니다.")

                            if "data" not in parsed_json_data or not isinstance(parsed_json_data["data"], dict):
                                raise ValueError("파싱된 데이터에 유효한 'data' 키가 없습니다.")

                            final_data = parsed_json_data["data"]

                            if "challenges" not in final_data:
                                raise ValueError("파싱된 데이터에 'challenges' 키가 없습니다.")

                            yield {
                                "event": "close",
                                "data": json.dumps({
                                    "status": 200,
                                    "message": "모든 챌린지 추천 완료",
                                    "data": final_data
                                }, ensure_ascii=False)
                            }
                            return
                            
                        except Exception as e:
                            yield {
                                "event": "error",
                                "data": json.dumps({
                                    "status": 500,
                                    "message": f"파싱 실패: {str(e)}",
                                    "data": None
                                }, ensure_ascii=False)
                            }

                    elif event_type == "error":
                        yield {
                            "event": "error",
                            "data": data_from_llm_model
                        }

                except Exception as e:
                    logger.error(f"이벤트 처리 중 에러 발생: {str(e)}")
                    yield {
                        "event": "error",
                        "data": json.dumps({
                            "status": 500,
                            "message": f"이벤트 처리 실패: {str(e)}",
                            "data": None
                        }, ensure_ascii=False)
                    }

        except Exception as e:
            logger.error(f"SSE 스트림 처리 중 에러 발생: {str(e)}")
            yield {
                "event": "error",
                "data": json.dumps({
                    "status": 500,
                    "message": f"서버 내부 오류로 추천에 실패했습니다: {str(e)}",
                    "data": None
                }, ensure_ascii=False)
            }
        finally:
            # 연결 종료 이벤트는 이미 파싱된 데이터와 함께 전송되었으므로 여기서는 전송하지 않음
            pass

    return EventSourceResponse(
        event_generator(),
        media_type="text/event-stream",
        ping= None 
    )

# LangChain 기반 RAG 추천
@router.get("/ai/chatbot/recommendation/free-text")
async def freetext_rag(
    sessionId: Optional[str] = Query(None),
    message: Optional[str] = Query(None)
):
    """
    자유 채팅 입력을 기반으로 친환경 챌린지를 추천하는 SSE 엔드포인트 (직접 generate + streamer)
    """
    # URL 디코딩 추가
    if message:
        message = unquote(message)
    # 입력값 검증
    if not message or not message.strip():
        return EventSourceResponse(
            event_generator_error("message는 필수입니다.", 400),
            media_type="text/event-stream"
        )
    if len(message.strip()) < 5:
        return EventSourceResponse(
            event_generator_error("message는 문자열이어야 하며, 최소 5자 이상의 문자열이어야 합니다.", 422),
            media_type="text/event-stream"
        )
    message_lower = message.lower()
    # 카테고리 관련 요청 체크
    category_reset_keywords = ["원래", "처음", "이전", "원래대로", "기존", "카테고리"]
    is_category_request = any(keyword in message_lower for keyword in category_reset_keywords)
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
            logger.info(f"Fallback triggered: {message}")
            yield {
                "event": "error",
                "data": json.dumps({
                    "status": 200,
                    "message": "저는 친환경 챌린지를 추천해드리는 Leafresh 챗봇이에요! 환경 관련 질문을 해주시면 더 잘 도와드릴 수 있어요.",
                    "data": None
                }, ensure_ascii=False)
            }
            return # fallback 메시지 전송 후 종료

        # 1. 카테고리 변경 로직 (키워드 기반)
        print(f"🚀FREE-TEXT ROUTER START 🚀")
        print(f"User message: {message}")
        print(f"Session ID: {sessionId}")
        
        current_category = conversation_states[sessionId].get("category", "제로웨이스트")
        message_lower = message.lower()
        
        # 카테고리 변경 검사
        category_changed = False
        print(f"🔍 키워드 검색 시작: '{message_lower}'")
        print(f"🔥 category_keywords 타입: {type(category_keywords)}")
        if isinstance(category_keywords, dict):
            for category, keywords in category_keywords.items():
                print(f"   - 카테고리 '{category}' 키워드 확인: {keywords}")
                if any(keyword in message_lower for keyword in keywords):
                    print(f"🎯 키워드 매칭 성공! '{category}' 카테고리로 변경")
                    conversation_states[sessionId]["category"] = category
                    current_category = category
                    category_changed = True
                    break
        else:
            print(f"category_keywords가 dict가 아님: {category_keywords}")
        
        if not category_changed:
            print(f"❌ 키워드 매칭 실패. 기존 카테고리 '{current_category}' 유지")
        
        # 2. context 추출 (RAG)
        messages_history = "\n".join(conversation_states[sessionId]["messages"])
        
        # RAG 검색 수행
        docs = retriever.get_relevant_documents(message)
        context = "\n".join([doc.page_content for doc in docs])
        logger.info(f"RAG 검색 완료. 문서 수: {len(docs)}")
        logger.info(f"컨텍스트 길이: {len(context)}")

        prompt = custom_prompt.format(
            context=context,  # RAG 활성화
            query=message,
            messages=messages_history,
            category=current_category
        )

        # LLM 응답 스트리밍
        try:
            full_response_text = ""
            for data_payload in get_free_text_llm_response(prompt, current_category):
                event_type = data_payload.get("event")
                data_from_llm_model = data_payload.get("data")

                if not event_type or not data_from_llm_model:
                    continue  # 유효하지 않은 이벤트는 건너뛰기

                # data_from_llm_model이 JSON 문자열인 경우 딕셔너리로 파싱 (키 접근 전)
                parsed_data_payload = json.loads(data_from_llm_model) if isinstance(data_from_llm_model, str) else data_from_llm_model

                if event_type == "challenge":
                    # 토큰 데이터를 추출하여 full_response_text에 누적
                    token_text = parsed_data_payload.get("data", "")
                    full_response_text += token_text

                    # 클라이언트에 토큰 전송
                    yield {
                        "event": "challenge",
                        "data": json.dumps({
                            "status": 200,
                            "message": "토큰 생성",
                            "data": token_text
                        }, ensure_ascii=False)
                    }

                elif event_type == "close":
                    # 대화 기록 업데이트 (LLM 최종 응답 전체를 저장)
                    if sessionId in conversation_states:
                        conversation_states[sessionId]["messages"].append(f"AI: {full_response_text}")

                    # 최종 응답 데이터 전송
                    yield {
                        "event": "close",
                        "data": json.dumps({
                            "status": 200,
                            "message": "모든 챌린지 추천 완료",
                            "data": parsed_data_payload.get("data", None)
                        }, ensure_ascii=False)
                    }
                    return

                elif event_type == "error":
                    yield {
                        "event": "error",
                        "data": data_from_llm_model
                    }

        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({
                    "status": 500,
                    "message": f"서버 내부 오류로 추천에 실패했습니다: {str(e)}",
                    "data": None
                }, ensure_ascii=False)
            }
            return

    return EventSourceResponse(
        event_generator(),
        media_type="text/event-stream",
        ping= None 
    )

async def event_generator_error(message: str, status_code: int):
    """에러 이벤트를 생성하는 제너레이터"""
    logger.error(f"SSE error event: {message} (status: {status_code})")
    yield {
        "event": "error",
        "data": json.dumps({
            "status": status_code,
            "message": message,
            "data": None
        }, ensure_ascii=False)
    }

# 명령어 예시 (vllm 모델 실행)
# python3 -m vllm.entrypoints.openai.api_server \
#   --model /home/ubuntu/mistral/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/e0bc86c23ce5aae1db576c8cca6f06f1f73af2db \
#   --host 0.0.0.0 \
#   --port 8800