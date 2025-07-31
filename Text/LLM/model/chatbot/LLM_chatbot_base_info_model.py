from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_qdrant import Qdrant
from langchain_community.embeddings import SentenceTransformerEmbeddings
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence, Optional, Dict, List, Generator, Any
from Text.LLM.model.chatbot.chatbot_constants import label_mapping, ENV_KEYWORDS, BAD_WORDS
from transformers import TextIteratorStreamer, LogitsProcessorList, InfNanRemoveLogitsProcessor
import torch
import os
import json
import random
import re
import threading
import logging
from fastapi import HTTPException
from fastapi.responses import JSONResponse
import gc
import unicodedata
import httpx

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

logger.info("Using shared Mistral model for base-info chatbot")

# base-info_response_schemas 정의
base_response_schemas = [
    ResponseSchema(name="recommend", description="추천 텍스트를 한글로 한 문장으로 출력해 주세요. (예: '이런 챌린지를 추천합니다.')"),
    ResponseSchema(name="challenges", description="추천 챌린지 리스트, 각 항목은 title, description 포함, description은 한글로 한 문장으로 요약해주세요.")
]

# base-info_output_parser 정의 
base_parser = StructuredOutputParser.from_response_schemas(base_response_schemas)

# base-info_prompt 정의 (단순한 한글 프롬프트)
base_prompt = PromptTemplate(
    input_variables=["location", "workType", "category"],
    template="""너는 챌린지 추천 챗봇이야. 사용자가 선택한 '위치, 직업, 카테고리'에 맞춰 구체적인 친환경 챌린지 3가지를 추천해줘.

위치: {location}
직업: {workType}
카테고리: {category}

중요한 요구사항:
- 반드시 올바른 JSON 객체만 출력해
- 모든 내용(recommend, title, description)은 반드시 한글 문장끝에는 "니다." 로만 작성해
- 각 챌린지는 "title"과 "description" 필드만 포함해                         
- title은 반드시 "1. ", "2. ", "3. " 형태로 번호를 붙여서 시작해
- description은 한 문장으로 간결하게 작성해 주세요.
- 영어, 이모지, 특수문자는 사용하지 마

출력 예시:
```json
{{
    "recommend": "이런 챌린지를 추천합니다.",
    "challenges": [
        {{"title": "첫번째 챌린지:", "description": "간단한 설명"}},
        {{"title": "두번째 챌린지:", "description": "간단한 설명"}},
        {{"title": "세번째 챌린지:", "description": "간단한 설명"}}
    ]
}}
```

반드시 위 예시와 같은 마크다운+JSON 구조로 한글로만 출력해. recommend는 한 문장, challenges는 3개 챌린지로!
"""
)

def get_llm_response(prompt: str, category: str) -> Generator[Dict[str, Any], None, None]:
    """vLLM 서버에 POST 요청하여 모델의 자연스러운 띄어쓰기를 포함한 응답을 SSE 형식으로 반환"""
    logger.info(f"[vLLM 호출] 프롬프트 길이: {len(prompt)}")
    url = "http://localhost:8800/v1/chat/completions"
    payload = {
        "model": "/home/ubuntu/mistral_finetuned_v5/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/0d4b76e1efeb5eb6f6b5e757c79870472e04bd3a",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": True,
        "max_tokens": 512,
        "temperature": 0.5,
        "do_sample": True
    }

    response_completed = False
    full_response = ""
    streaming_buffer = ""  # 스트리밍을 위한 버퍼
    
    recommend_sentence_finished = False 
    full_cleaned_text_stream = "" 

    try:
        with httpx.stream("POST", url, json=payload, timeout=60.0) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data: "):
                    json_str = line[len("data: "):]
                    if json_str.strip() == "[DONE]":
                        break
                    
                    try:
                        json_data = json.loads(json_str)
                        delta = json_data["choices"][0]["delta"]
                        token = delta.get("content", "")
                        
                        if not token:
                            continue
                        
                        logger.info(f"토큰 수신: {token[:20]}...")

                        full_response += token
                        streaming_buffer += token
                        
                        if ' ' in streaming_buffer:
                            parts = streaming_buffer.rsplit(' ', 1)
                            to_flush = parts[0] + ' '
                            streaming_buffer = parts[1]

                            if to_flush:
                                cleaned_text = to_flush
                                cleaned_text = re.sub(r'"(recommend|challenges|title|description)":\s*("|\')?', '', cleaned_text)
                                cleaned_text = cleaned_text.replace("```json", "").replace("```", "")
                                cleaned_text = re.sub(r'["\']', '', cleaned_text)
                                cleaned_text = re.sub(r'[\[\]{}$]', '', cleaned_text)
                                cleaned_text = re.sub(r',\s*$', '', cleaned_text)
                                
                                if cleaned_text.strip() and not response_completed:
                                    # [변경사항 주석] 2024-07-30
                                    # 각 챌린지 항목이 이어서 출력되는 문제를 해결하기 위해,
                                    # 새로운 챌린지 번호(2. 또는 3.)가 시작되기 전에 줄바꿈을 추가하는 로직입니다.
                                    challenge_start_match = re.search(r'(2\.|3\.)', cleaned_text)
                                    if challenge_start_match:
                                        start_index = challenge_start_match.start()
                                        part_before_challenge = cleaned_text[:start_index]
                                        part_after_challenge = cleaned_text[start_index:]

                                        # 1. 새 챌린지 번호 이전의 텍스트를 먼저 전송합니다.
                                        if part_before_challenge.strip():
                                            yield {"event": "challenge", "data": json.dumps({"status": 200, "message": "토큰 생성", "data": part_before_challenge}, ensure_ascii=False)}
                                        
                                        # 2. 챌린지를 구분하기 위한 줄바꿈을 전송합니다.
                                        yield {"event": "challenge", "data": json.dumps({"status": 200, "message": "토큰 생성", "data": "\n\n"}, ensure_ascii=False)}
                                        
                                        # 3. 새 챌린지 번호와 그 이후의 텍스트를 전송합니다.
                                        yield {"event": "challenge", "data": json.dumps({"status": 200, "message": "토큰 생성", "data": part_after_challenge}, ensure_ascii=False)}
                                        
                                        # 누적 텍스트 버퍼에도 반영합니다.
                                        full_cleaned_text_stream += cleaned_text

                                    else:
                                        # 일반적인 경우, 텍스트를 그대로 전송합니다.
                                        yield {
                                            "event": "challenge",
                                            "data": json.dumps({
                                                "status": 200,
                                                "message": "토큰 생성",
                                                "data": cleaned_text
                                            }, ensure_ascii=False)
                                        }
                                        full_cleaned_text_stream += cleaned_text
                                    
                                    # recommend 문장 완성 후 줄바꿈을 추가하는 로직은 그대로 유지합니다.
                                    recommend_endings = ["추천합니다.", "추천드려요.", "추천해요.", "권장합니다."]
                                    if not recommend_sentence_finished and any(full_cleaned_text_stream.strip().endswith(ending) for ending in recommend_endings):
                                        yield {
                                            "event": "challenge",
                                            "data": json.dumps({
                                                "status": 200,
                                                "message": "토큰 생성",
                                                "data": "\n\n"
                                            }, ensure_ascii=False)
                                        }
                                        recommend_sentence_finished = True
                                
                    except json.JSONDecodeError:
                        logger.warning(f"JSON 디코딩 실패: {json_str}")
                        continue

            # 스트리밍 루프가 끝난 후 버퍼에 남은 마지막 텍스트 조각을 정제하여 전송합니다.
            if streaming_buffer:
                cleaned_text = streaming_buffer
                cleaned_text = re.sub(r'"(recommend|challenges|title|description)":\s*("|\')?', '', cleaned_text)
                cleaned_text = cleaned_text.replace("```json", "").replace("```", "")
                cleaned_text = re.sub(r'["\']', '', cleaned_text)
                cleaned_text = re.sub(r'[\[\]{}$]', '', cleaned_text)
                cleaned_text = re.sub(r',\s*$', '', cleaned_text)
                
                if cleaned_text.strip() and not response_completed:
                    yield {
                        "event": "challenge",
                        "data": json.dumps({
                            "status": 200,
                            "message": "토큰 생성",
                            "data": cleaned_text.strip()
                        }, ensure_ascii=False)
                    }
                    
                    full_cleaned_text_stream += cleaned_text.strip()
                    recommend_endings = ["추천합니다.", "추천드려요.", "추천해요.", "권장합니다."]
                    
                    if not recommend_sentence_finished and any(full_cleaned_text_stream.strip().endswith(ending) for ending in recommend_endings):
                        yield {
                            "event": "challenge",
                            "data": json.dumps({
                                "status": 200,
                                "message": "토큰 생성",
                                "data": "\n\n"
                            }, ensure_ascii=False)
                        }
                        recommend_sentence_finished = True

        # --- [최종 파싱 로직] ---
        # 스트리밍이 모두 끝난 후, 전체 응답(full_response)을 정리하고 파싱합니다.
        logger.info("스트리밍 완료. 전체 응답 파싱 시작.")
        
        match = re.search(r'```json\s*([\s\S]*?)\s*```', full_response)
        if match:
            json_str = match.group(1)
        else:
            start_idx = full_response.find('{')
            end_idx = full_response.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = full_response[start_idx:end_idx]
            else:
                raise ValueError("응답에서 유효한 JSON 객체를 찾을 수 없습니다.")
        
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)

        try:
            parsed_data = json.loads(json_str)
            
            eng_label = label_mapping.get(category, "etc")
            if isinstance(parsed_data, dict) and "challenges" in parsed_data:
                for challenge in parsed_data["challenges"]:
                    challenge["category"] = eng_label
            
            if not response_completed:
                response_completed = True
                yield {
                    "event": "close",
                    "data": json.dumps({
                        "status": 200,
                        "message": "모든 챌린지 추천 완료",
                        "data": parsed_data
                    }, ensure_ascii=False)
                }
        except json.JSONDecodeError as e:
            logger.error(f"최종 JSON 파싱 실패: {e}")
            logger.error(f"파싱 시도한 문자열: {json_str}")
            if not response_completed:
                response_completed = True
                yield {
                    "event": "error",
                    "data": json.dumps({
                        "status": 500,
                        "message": f"JSON 파싱 실패: {e}",
                        "data": full_response
                    }, ensure_ascii=False)
                }

    except httpx.HTTPStatusError as e:
        logger.error(f"vLLM 서버 오류: {e.response.status_code} - {e.response.text}")
        if not response_completed:
            yield {"event": "error", "data": json.dumps({"status": 500, "message": f"vLLM 서버 오류: {e.response.text}"})}
    except Exception as e:
        logger.error(f"[vLLM 호출 실패] {str(e)}")
        if not response_completed:
            yield {"event": "error", "data": json.dumps({"status": 500, "message": f"vLLM 호출 실패: {str(e)}"}, ensure_ascii=False)}
