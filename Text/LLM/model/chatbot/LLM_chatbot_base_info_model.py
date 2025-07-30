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
``` 
    ```json
    {{
        "recommend": "이런 챌린지를 추천합니다.",
        "challenges": [
            {{"title": "첫번째 챌린지:",
             "description": "간단한 설명"}},
            {{"title": "두번째 챌린지:",
             "description": "간단한 설명"}},
            {{"title": "세번째 챌린지:",
             "description": "간단한 설명"}}
        ]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
    }}
    ```
```

반드시 위 예시와 같은 마크다운+JSON 구조로 한글로만 출력해. recommend는 한 문장, challenges는 3개 챌린지로!
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
        "model": "/home/ubuntu/mistral_finetuned_v5/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/0d4b76e1efeb5eb6f6b5e757c79870472e04bd3a",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": True,
        "max_tokens": 512,
        "temperature": 0.5,
        "do_sample": True # temperature 설정 시 반드시 True로 설정해야 함: 확률적 샘플링 활성화
    }

    response_completed = False  # 응답 완료 여부를 추적하는 플래그
    token_buffer = ""  # 토큰을 누적할 버퍼
    full_sentence = ""  # 누적 문장 버퍼 (추천문장 등 문장 단위 줄바꿈 플러시용)
    # 한글과 영어 모두를 고려한 단어 구분자 (줄바꿈 포함)
    word_delimiters = [' ', '\t', '\n', '.', ',', '!', '?', ';', ':', '"', "'", '(', ')', '[', ']', '{', '}', '<', '>', '/','~', '은', '는', '가', '을', '를', '의', '에서', '으로', '와', '과', '만', '부터', '까지','든지', '라도', '으로', '께서', '분들께', '고', '며', '면', '거나', '든가', '위해', '위한', '도시에서', '바닷가에서', '산에서', '농촌에서', '사무직', '현장직', '영업직', '재택근무']

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
                                # 구분자 기준으로 token_buffer 나누기
                                words = []
                                current_word = ""
                                i = 0
                                while i < len(token_buffer):
                                    char = token_buffer[i]
                                    # "."만 뒤에 단어와 함께 flush, 나머지 구분자는 앞에 단어와 함께 flush
                                    if char == '.':
                                        if current_word:
                                            words.append(current_word)
                                            current_word = ""
                                        current_word += char  # "."로 새 단어 시작 (뒤에 단어와 함께)
                                    elif char in word_delimiters:
                                        if current_word:
                                            words.append(current_word + char)  # 구분자를 앞에 단어와 함께
                                            current_word = ""
                                        else:
                                            current_word += char  # 단독 구분자인 경우
                                    else:
                                        current_word += char
                                    i += 1 # 다음 문자로 이동
                                if current_word:
                                    words.append(current_word)

                                # \n delimiter는 그대로 전송
                                final_words = []
                                for word in words:
                                    if word.strip() == '\n':
                                        final_words.append('\n')
                                    else:
                                        final_words.append(word)

                                # 최종 단어 리스트가 1개 이상인 경우, 마지막 단어를 제외한 나머지 단어들을 처리
                                if len(final_words) > 1:
                                    complete_words = final_words[:-1]
                                    token_buffer = final_words[-1] if final_words else ""
                                    for word in complete_words:
                                        # 번호 구분자가 앞에 있는 경우, 공백 삽입 (예: '1.챌린지' → '1. 챌린지')
                                        word = re.sub(r"^(\d+\.)\s*(?=\S)", r"\1 ", word)
                                        # \n 문자는 그대로 전송
                                        if word == '\n':
                                            yield {
                                                "event": "challenge",
                                                "data": json.dumps({
                                                    "status": 200,
                                                    "message": "토큰 생성",
                                                    "data": "\n"
                                                }, ensure_ascii=False)
                                            }
                                            continue

                                        # 토큰 정제 - 순수 텍스트만 추출
                                        cleaned_text = word
                                        # JSON 관련 문자열 제거
                                        cleaned_text = re.sub(r'"(recommend|challenges|title|description)":\s*("|\')?', '', cleaned_text)
                                        # 마크다운 및 JSON 구조 제거
                                        cleaned_text = cleaned_text.replace("```json", "").replace("```", "").strip()
                                        cleaned_text = re.sub(r'["\']', '', cleaned_text)  # 따옴표 제거
                                        cleaned_text = re.sub(r'[\[\]{}$]', '', cleaned_text)  # 괄호와 $ 제거
                                        cleaned_text = re.sub(r',\s*$', '', cleaned_text)  # 끝의 쉼표 제거
                                        # 줄바꿈 보존: 이스케이프된 줄바꿈을 실제 줄바꿈으로 변환
                                        cleaned_text = cleaned_text.replace('\\\\n', '\n')  # 이중 이스케이프된 줄바꿈을 실제 줄바꿈으로 변환
                                        cleaned_text = cleaned_text.replace('\\n', '\n')  # 이스케이프된 줄바꿈을 실제 줄바꿈으로 변환
                                        # 줄바꿈이 아닌 백슬래시만 제거
                                        cleaned_text = cleaned_text.replace('\\\\', '')  # 이중 백슬래시 제거 (줄바꿈 제외)
                                        # \n을 제외한 다른 이스케이프 문자들만 제거
                                        cleaned_text = re.sub(r'\\(?!n)', '', cleaned_text)  # \n을 제외한 백슬래시 제거
                                        # 추가: 연속된 공백을 하나로 정리하되 줄바꿈은 보존
                                        cleaned_text = re.sub(r' +', ' ', cleaned_text)  # 공백만 정리

                                        cleaned_text = cleaned_text.strip()
                                        if cleaned_text and cleaned_text.strip() not in ["", "``", "```"] and not response_completed:
                                            # 문자열 \n을 실제 줄바꿈으로 변환 처리
                                            if '\\n' in cleaned_text:
                                                parts = cleaned_text.split('\\n')
                                                for i, part in enumerate(parts):
                                                    if part.strip():
                                                        yield {
                                                            "event": "challenge",
                                                            "data": json.dumps({
                                                                "status": 200,
                                                                "message": "토큰 생성",
                                                                "data": part.strip()
                                                            }, ensure_ascii=False)
                                                        }
                                                    if i < len(parts) - 1:  # 마지막이 아닌 경우 줄바꿈 추가
                                                        yield {
                                                            "event": "challenge",
                                                            "data": json.dumps({
                                                                "status": 200,
                                                                "message": "토큰 생성",
                                                                "data": "\n"
                                                            }, ensure_ascii=False)
                                                        }
                                            else:
                                                # 줄바꿈 로직 개선: 추천 문장에 대해 두 줄, 일반 문장에 대해 한 줄
                                                recommend_endings = ["추천합니다.", "추천드려요.", "추천해요.", "권장합니다."]
                                                full_sentence += cleaned_text
                                                if any(full_sentence.strip().endswith(ending) for ending in recommend_endings):
                                                    cleaned_text += '\n\n'
                                                    full_sentence = ""
                                                elif cleaned_text.endswith(".") or cleaned_text.endswith("세요") or cleaned_text.endswith("니다") or cleaned_text.endswith("합니다"):
                                                    cleaned_text += '\n\n'
                                                    full_sentence = ""
                                                # 최종 출력 전에 \n을 실제 줄바꿈으로 변환
                                                final_text = cleaned_text.replace('\\n', '\n')
                                                yield {
                                                    "event": "challenge",
                                                    "data": json.dumps({
                                                        "status": 200,
                                                        "message": "토큰 생성",
                                                        "data": final_text #단어 단위 출력
                                                    }, ensure_ascii=False)
                                                }
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
                                # Split token_buffer by delimiters
                                words = []
                                current_word = ""
                                i = 0
                                while i < len(token_buffer):
                                    char = token_buffer[i]
                                    # "."만 뒤에 단어와 함께 flush, 나머지 구분자는 앞에 단어와 함께 flush
                                    if char == '.':
                                        if current_word:
                                            words.append(current_word)
                                            current_word = ""
                                        current_word += char  # "."로 새 단어 시작 (뒤에 단어와 함께)
                                    elif char in word_delimiters:
                                        if current_word:
                                            words.append(current_word + char)  # 구분자를 앞에 단어와 함께
                                            current_word = ""
                                        else:
                                            current_word += char  # 단독 구분자인 경우
                                    else:
                                        current_word += char
                                    i += 1
                                if current_word:
                                    words.append(current_word)

                                # For newline delimiters, keep them as standalone entries
                                final_words = []
                                for word in words:
                                    if word.strip() == '\n':
                                        final_words.append('\n')
                                    else:
                                        final_words.append(word)

                                if len(final_words) > 1:
                                    complete_words = final_words[:-1]
                                    token_buffer = final_words[-1] if final_words else ""
                                    for word in complete_words:
                                        # 번호 구분자가 앞에 있는 경우, 공백 삽입 (예: '1.챌린지' → '1. 챌린지')
                                        word = re.sub(r"^(\d+\.)\s*(?=\S)", r"\1 ", word)
                                        # \n 문자는 그대로 전송 (줄바꿈 처리)
                                        if word == '\n':
                                            yield {
                                                "event": "challenge",
                                                "data": json.dumps({
                                                    "status": 200,
                                                    "message": "토큰 생성",
                                                    "data": "\n"
                                                }, ensure_ascii=False)
                                            }
                                            continue

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
                                        # 콜론만 단독, 콜론+공백류, 콜론+줄바꿈 등도 필터링
                                        if re.fullmatch(r":\s*", cleaned_text) or cleaned_text in ["json", "recommend", "challenges", "title", "description"]:
                                            continue
                                        if cleaned_text and cleaned_text.strip() not in ["", "``", "```"] and not response_completed:
                                            # 줄바꿈 로직 개선: 추천 문장에 대해 두 줄, 일반 문장에 대해 한 줄
                                            recommend_endings = ["추천합니다.", "추천드려요.", "추천해요.", "권장합니다."]
                                            full_sentence += cleaned_text
                                            if any(full_sentence.strip().endswith(ending) for ending in recommend_endings):
                                                cleaned_text += '\n\n'
                                                full_sentence = ""
                                            elif cleaned_text.endswith(".") or cleaned_text.endswith("세요.") or cleaned_text.endswith("니다.") or cleaned_text.endswith("합니다."):
                                                cleaned_text += '\n\n'
                                                full_sentence = ""
                                            # 최종 출력 전에 \n을 실제 줄바꿈으로 변환
                                            final_text = cleaned_text.replace('\\n', '\n')
                                            yield {
                                                "event": "challenge",
                                                "data": json.dumps({
                                                    "status": 200,
                                                    "message": "토큰 생성",
                                                    "data": final_text
                                                }, ensure_ascii=False)
                                            }
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