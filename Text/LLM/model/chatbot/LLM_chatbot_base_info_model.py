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
# from Text.LLM.model.chatbot.shared_model import shared_model

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

# # 공유 모델 사용(vLLM 사용으로 인해 주석처리)
# model = shared_model.model
# tokenizer = shared_model.tokenizer

logger.info("Using shared Mistral model for base-info chatbot")

# base-info_response_schemas 정의
base_response_schemas = [
    ResponseSchema(name="recommend", description="추천 텍스트를 한글로 한 문장으로 출력해 주세요. (예: '이런 챌린지를 추천합니다.')"),
    ResponseSchema(name="challenges", description="추천 챌린지 리스트, 각 항목은 title, description 포함, description은 한글로 한 문장으로 요약해주세요.")
]

# base-info_output_parser 정의 
base_parser = StructuredOutputParser.from_response_schemas(base_response_schemas)

# base-info_prompt 정의
escaped_format = base_parser.get_format_instructions().replace("{", "{{").replace("}", "}}")
base_prompt = PromptTemplate(
    input_variables=["location", "workType", "category"],
    template=f"""당신은 환경 보호 챌린지를 추천하는 AI 어시스턴트입니다.
{{location}}의 환경에 있는 {{workType}} 사용자가 {{category}}를 실천할 때,
절대적으로 환경에 도움이 되는 챌린지를 아래 JSON 형식으로 3가지 추천해주세요.

주의사항:
1. 모든 속성 이름과 문자열 값은 반드시 큰따옴표(")로 둘러싸야 합니다.
2. recommend 필드에는 {{category}} 관련 추천 문구를 포함해야 합니다.
3. 각 title 내용은 번호를 붙이세요.
4. description은 한 문장으로만 작성하세요. (50자 이내)

JSON 포맷:
{escaped_format}

반드시 위 JSON 형식 그대로 반드시 한글로 한번만 출력하세요.
"""
)

def format_sse_response(event: str, data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "event": event,
        "data": json.dumps(data, ensure_ascii=False)
    }


# vLLM 서버 호출용 httpx 사용
import httpx

def get_llm_response(prompt: str, category: str) -> Generator[Dict[str, Any], None, None]:
    """vLLM 서버에 POST 요청하여 응답을 SSE 형식으로 반환"""
    logger.info(f"[vLLM 호출] 프롬프트 길이: {len(prompt)}")
    url = "http://localhost:8800/v1/chat/completions"
    payload = {
        "model": "/home/ubuntu/mistral/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/e0bc86c23ce5aae1db576c8cca6f06f1f73af2db",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": 2048
    }

    response_completed = False  # 응답 완료 여부를 추적하는 플래그
    token_buffer = ""  # 토큰을 누적할 버퍼
    # 한글과 영어 모두를 고려한 단어 구분자 (줄바꿈 제외)
    word_delimiters = [' ', '\t', '.', ',', '!', '?', ';', ':', '"', "'", '(', ')', '[', ']', '{', '}', '<', '>', '/', '|', '&', '*', '+', '-', '=', '_', '@', '#', '$', '%', '^', '~', '`', '은', '는', '이', '가', '을', '를', '의', '에', '에서', '로', '으로', '와', '과', '도', '만', '부터', '까지', '나', '든지', '라도', '라서', '고', '며', '거나', '든가', '든']

    try:
        with httpx.stream("POST", url, json=payload, timeout=60.0) as response:
            full_response = ""
            for line in response.iter_lines():
                # line이 bytes인지 str인지 체크해서 맞는 타입으로 비교
                if isinstance(line, bytes):
                    if line.startswith(b"data: "):
                        try:
                            json_data = json.loads(line[len(b"data: "):])
                            delta = json_data["choices"][0]["delta"]
                            token = delta.get("content", "") #vLLM에서 한글자씩 받음
                            if token.strip() in ["```", "`", ""]:
                                continue  # 이런 토큰은 누적하지 않음
                            full_response += token
                            token_buffer += token # 토큰 버퍼에 한글자씩 누적
                            logger.info(f"토큰 수신: {token[:20]}...")

                            # 토큰 버퍼에서 단어 단위로 분리하여 스트리밍
                            if any(delimiter in token_buffer for delimiter in word_delimiters):
                                # 단어 경계를 찾아서 분리
                                words = []
                                current_word = ""
                                for char in token_buffer:
                                    if char in word_delimiters:
                                        if current_word:
                                            words.append(current_word)
                                            current_word = ""
                                        words.append(char)
                                    else:
                                        current_word += char
                                
                                if current_word:
                                    words.append(current_word)
                                
                                # 완성된 단어들만 스트리밍하고, 마지막 불완전한 단어는 버퍼에 유지
                                if len(words) > 1:
                                    # 마지막 단어가 불완전할 수 있으므로 제외
                                    complete_words = words[:-1] # 완성된 단어들
                                    token_buffer = words[-1] if words else "" # 마지막 불완전한 단어 버퍼에 유지
                                    
                                    for word in complete_words: # 완성된 단어들을 프론트엔드로 전송
                                        # 토큰 정제 - 순수 텍스트만 추출
                                        cleaned_text = word
                                        # JSON 관련 문자열 제거
                                        cleaned_text = re.sub(r'"(recommend|challenges|title|description)":\s*("|\')?', '', cleaned_text)
                                        # 마크다운 및 JSON 구조 제거
                                        cleaned_text = cleaned_text.replace("```json", "").replace("```", "").strip()
                                        cleaned_text = re.sub(r'["\']', '', cleaned_text)  # 따옴표 제거
                                        cleaned_text = re.sub(r'[\[\]{}$]', '', cleaned_text)  # 괄호와 $ 제거
                                        cleaned_text = re.sub(r',\s*$', '', cleaned_text)  # 끝의 쉼표 제거
                                        # 줄바꿈 보존: \n은 그대로 두고 다른 공백 문자만 제거
                                        cleaned_text = re.sub(r'[ \t\r\f\v]+', ' ', cleaned_text)  # \n 제외 공백만 제거
                                        # 이스케이프된 문자들을 실제 문자로 변환
                                        cleaned_text = cleaned_text.replace('\\\\n', '\n')  # 이중 이스케이프된 줄바꿈을 실제 줄바꿈으로 변환
                                        cleaned_text = cleaned_text.replace('\\n', '\n')  # 이스케이프된 줄바꿈을 실제 줄바꿈으로 변환
                                        # 백슬래시 제거 (줄바꿈이 아닌 경우)
                                        cleaned_text = cleaned_text.replace('\\\\', '')  # 이중 백슬래시 제거
                                        cleaned_text = cleaned_text.replace('\\', '')  # 단일 백슬래시 제거
                                        # 추가: 연속된 공백을 하나로 정리하되 줄바꿈은 보존
                                        cleaned_text = re.sub(r' +', ' ', cleaned_text)  # 공백만 정리
                                        
                                        cleaned_text = cleaned_text.strip()
                                        if cleaned_text and cleaned_text.strip() not in ["", "``", "```"] and not response_completed:
                                            # title 내용이 끝날 때 줄바꿈 추가 (번호.으로 끝나는 경우)
                                            if re.search(r'\d+\.$', cleaned_text) or cleaned_text.endswith('title') or cleaned_text.endswith('description'):
                                                cleaned_text += '\n'
                                            yield {
                                                "event": "challenge",
                                                "data": json.dumps({
                                                    "status": 200,
                                                    "message": "토큰 생성",
                                                    "data": cleaned_text #단어 단위 출력
                                                }, ensure_ascii=False)
                                            }
                                else:
                                    # 단어가 하나뿐이면 버퍼에 유지
                                    pass
                        except Exception as e:
                            logger.error(f"[vLLM 토큰 파싱 실패] {str(e)}")
                            continue
                elif isinstance(line, str):
                    if line.startswith("data: "):
                        try:
                            json_data = json.loads(line[len("data: "):])
                            delta = json_data["choices"][0]["delta"]
                            token = delta.get("content", "")
                            if token.strip() in ["```", "`", ""]:
                                continue  # 이런 토큰은 누적하지 않음
                            full_response += token
                            token_buffer += token
                            logger.info(f"토큰 수신: {token[:20]}...")

                            # 토큰 버퍼에서 단어 단위로 분리하여 스트리밍
                            if any(delimiter in token_buffer for delimiter in word_delimiters):
                                # 단어 경계를 찾아서 분리
                                words = []
                                current_word = ""
                                for char in token_buffer:
                                    if char in word_delimiters:
                                        if current_word:
                                            words.append(current_word)
                                            current_word = ""
                                        words.append(char)
                                    else:
                                        current_word += char
                                
                                if current_word:
                                    words.append(current_word)
                                
                                # 완성된 단어들만 스트리밍하고, 마지막 불완전한 단어는 버퍼에 유지
                                if len(words) > 1:
                                    # 마지막 단어가 불완전할 수 있으므로 제외
                                    complete_words = words[:-1]
                                    token_buffer = words[-1] if words else ""
                                    
                                    for word in complete_words:
                                        # 토큰 정제 - 순수 텍스트만 추출
                                        cleaned_text = word
                                        # JSON 관련 문자열 제거
                                        cleaned_text = re.sub(r'"(recommend|challenges|title|description)":\s*("|\')?', '', cleaned_text)
                                        # 마크다운 및 JSON 구조 제거
                                        cleaned_text = cleaned_text.replace("```json", "").replace("```", "").strip()
                                        cleaned_text = re.sub(r'["\']', '', cleaned_text)  # 따옴표 제거
                                        cleaned_text = re.sub(r'[\[\]{}$]', '', cleaned_text)  # 괄호와 $ 제거
                                        cleaned_text = re.sub(r',\s*$', '', cleaned_text)  # 끝의 쉼표 제거
                                        # 줄바꿈 보존: \n은 그대로 두고 다른 공백 문자만 제거
                                        cleaned_text = re.sub(r'[ \t\r\f\v]+', ' ', cleaned_text)  # \n 제외 공백만 제거
                                        # 이스케이프된 문자들을 실제 문자로 변환
                                        cleaned_text = cleaned_text.replace('\\\\n', '\n')  # 이중 이스케이프된 줄바꿈을 실제 줄바꿈으로 변환
                                        cleaned_text = cleaned_text.replace('\\n', '\n')  # 이스케이프된 줄바꿈을 실제 줄바꿈으로 변환
                                        # 백슬래시 제거 (줄바꿈이 아닌 경우)
                                        cleaned_text = cleaned_text.replace('\\\\', '')  # 이중 백슬래시 제거                                        cleaned_text = cleaned_text.replace('\\\', '')  # 
                                        cleaned_text = cleaned_text.replace('\\', '')  # 단일 백슬래시 제거
                                        # 추가: 연속된 공백을 하나로 정리하되 줄바꿈은 보존
                                        cleaned_text = re.sub(r' +', ' ', cleaned_text)  # 공백만 정리
                                        
                                        cleaned_text = cleaned_text.strip()
                                        if cleaned_text and cleaned_text.strip() not in ["", "``", "```"] and not response_completed:
                                            # title 내용이 끝날 때 줄바꿈 추가 (번호.으로 끝나는 경우)
                                            if re.search(r'\d+\.$', cleaned_text) or cleaned_text.endswith('title') or cleaned_text.endswith('description'):
                                                cleaned_text += '\n'
                                            yield {
                                                "event": "challenge",
                                                "data": json.dumps({
                                                    "status": 200,
                                                    "message": "토큰 생성",
                                                    "data": cleaned_text
                                                }, ensure_ascii=False)
                                            }
                                else:
                                    # 단어가 하나뿐이면 버퍼에 유지
                                    pass
                        except Exception as e:
                            logger.error(f"[vLLM 토큰 파싱 실패] {str(e)}")
                            continue
        # 최종 JSON 파싱 시도
        try:
            logger.info("스트리밍 완료. 전체 응답 파싱 시작.")
            json_str = full_response.strip()
            # 코드블록, 백틱 등 제거
            json_str = json_str.replace("```json", "").replace("```", "").replace("`", "").strip()
            # 깨진 유니코드 제거 (예: \udc80 등)
            json_str = json_str.encode("utf-8", "ignore").decode("utf-8", "ignore")
            # 제어 문자 제거 (줄바꿈 제외)
            json_str = ''.join(c for c in json_str if unicodedata.category(c)[0] != 'C' or c == '\n')
            # 이스케이프된 문자들을 실제 문자로 변환
            json_str = json_str.replace('\\\n', '\n')  # 이중 이스케이프된 줄바꿈을 실제 줄바꿈으로 변환
            json_str = json_str.replace('\\n', '\n')  # 이스케이프된 줄바꿈을 실제 줄바꿈으로 변환
            # 백슬래시 제거 (줄바꿈이 아닌 경우)
            json_str = json_str.replace('\\\\', '')  # 이중 백슬래시 제거
            json_str = json_str.replace('\\', '')  # 단일 백슬래시 제거
            # 중복 JSON 제거 - 첫 번째 완전한 JSON만 추출
            json_objects = []
            brace_count = 0
            start_idx = -1
            for i, char in enumerate(json_str):
                if char == '{':
                    if brace_count == 0:
                        start_idx = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and start_idx != -1:
                        json_obj = json_str[start_idx:i+1]
                        try:
                            json.loads(json_obj)
                            json_objects.append(json_obj)
                            break
                        except:
                            continue
            if json_objects:
                json_str = json_objects[0]
            else:
                if "{" in json_str and "}" in json_str:
                    json_str = json_str[json_str.find("{"):json_str.rfind("}")+1]
            json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
            json_str = re.sub(r',\s*,', ',', json_str)
            # 줄바꿈은 보존하면서 다른 공백만 정리
            json_str = re.sub(r'[ \t\r\f\v]+', ' ', json_str)
            logger.info(f"파싱 시도 문자열: {json_str}")
            
            # JSON 파싱 시도
            try:
                parsed_temp = json.loads(json_str)
                # base_parser.parse() 안전하게 처리
                try:
                    parsed_data = base_parser.parse(json.dumps(parsed_temp))
                except Exception as parse_error:
                    logger.error(f"base_parser.parse() 실패: {str(parse_error)}")
                    # fallback: 기본 구조로 변환
                    if isinstance(parsed_temp, dict):
                        parsed_data = {
                            "recommend": parsed_temp.get("recommend", "챌린지를 추천합니다."),
                            "challenges": parsed_temp.get("challenges", [])
                        }
                    else:
                        # 완전한 fallback
                        parsed_data = {
                            "recommend": "챌린지를 추천합니다.",
                            "challenges": []
                        }
                
                # 카테고리 정보 추가 (줄바꿈 강제 추가 로직 제거)
                eng_label = label_mapping[category]
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
            except json.JSONDecodeError as json_error:
                logger.error(f"JSON 파싱 실패: {str(json_error)}")
                logger.error(f"파싱 시도한 문자열: {json_str}")
                
                # 이스케이프된 JSON 문자열 처리 시도
                try:
                    # "json{...}" 형태 처리
                    if json_str.startswith('"json{') and json_str.endswith('}"'):
                        inner_json = json_str[6:-2]  # "json{" 와 "}" 제거
                        # 이스케이프된 문자들 처리
                        inner_json = inner_json.replace('\\"', '"').replace('\\\\', '\\')
                        parsed_temp = json.loads(inner_json)
                        if isinstance(parsed_temp, dict):
                            parsed_data = {
                                "recommend": parsed_temp.get("recommend", "챌린지를 추천합니다."),
                                "challenges": parsed_temp.get("challenges", [])
                            }
                        else:
                            raise ValueError("내부 JSON이 딕셔너리가 아님")
                    else:
                        raise ValueError("지원하지 않는 JSON 형식")
                except Exception as inner_error:
                    logger.error(f"이스케이프된 JSON 처리 실패: {str(inner_error)}")
                    # 완전한 fallback
                    parsed_data = {
                        "recommend": full_response.strip(),
                        "challenges": []
                    }
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
            except Exception as parse_error:
                logger.error(f"파싱 중 예상치 못한 오류: {str(parse_error)}")
                # 완전한 fallback
                parsed_data = {
                    "recommend": full_response.strip(),
                    "challenges": []
                }
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
                    
        except Exception as e:
            logger.error(f"[vLLM 파싱 실패] {str(e)}")
            logger.error(f"원본 응답: {full_response[:500]}...")
            if not response_completed:
                response_completed = True
                yield {
                    "event": "error",
                    "data": json.dumps({
                        "status": 500,
                        "message": f"JSON 파싱 실패: {str(e)}",
                        "data": None
                    }, ensure_ascii=False)
                }
    except Exception as e:
        logger.error(f"[vLLM 호출 실패] {str(e)}")
        if not response_completed:
            response_completed = True
            yield {
                "event": "error",
                "data": json.dumps({
                    "status": 500,
                    "message": f"vLLM 호출 실패: {str(e)}",
                    "data": None
                }, ensure_ascii=False)
            }
