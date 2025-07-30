from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_qdrant import Qdrant
from langchain_community.embeddings import SentenceTransformerEmbeddings
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence, Optional, Dict, List, Generator, Any
from Text.LLM.model.chatbot.chatbot_constants import label_mapping, ENV_KEYWORDS, BAD_WORDS, category_keywords
from transformers import TextIteratorStreamer, LogitsProcessorList, InfNanRemoveLogitsProcessor
import torch
import os
import json
import random
import re
import threading
import logging
from fastapi import HTTPException
import gc
import unicodedata
# from Text.LLM.model.chatbot.shared_model import shared_model
import httpx

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

logger.info("Using shared Mistral model for free-text chatbot")

# Qdrant 설정
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embedding_model = SentenceTransformerEmbeddings(model_name="BAAI/bge-small-en-v1.5")

vectorstore = Qdrant(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embeddings=embedding_model
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# RAG 방식 챌린지 추천을 위한 Output Parser 정의
rag_response_schemas = [
    ResponseSchema(name="recommend", description="추천 텍스트를 한글로 한 문장으로 출력해주세요. (예: '이런 챌린지를 추천합니다.')"),
    ResponseSchema(name="challenges", description="추천 챌린지 리스트, 각 항목은 title, description 포함, description은 한글로 한 문장으로 요약해주세요.")
]

# LangChain의 StructuredOutputParser를 사용하여 JSON 포맷을 정의
rag_parser = StructuredOutputParser.from_response_schemas(rag_response_schemas)

# chat_template 단순화에 맞춘 프롬프트 (단순화)
custom_prompt = PromptTemplate(
    input_variables=["context", "query", "messages", "category"],
    template="""
너는 사용자와 자유롭게 대화하며 대화의 맥락에 맞는 친환경 챌린지 3가지를 JSON 형식으로 추천하는 챗봇이야.

아래 참고 문서와 이전 대화를 바탕으로 사용자의 상황에 맞는 친환경 챌린지 3가지를 추천해줘.

참고 문서:
{context}

이전 대화:
{messages}

현재 카테고리: {category}
사용자 질문: {query}

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
    "recommend": "사용자 상황에 맞는 한 문장 추천 텍스트",

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

반드시 위 예시와 같은 마크다운+JSON 구조로 한글로만 출력해. recommend는 한 문장, challenges는 3개 챌린지로! 출력 형식은 반드시 마크다운+JSON 구조로!
"""
)

def get_llm_response(prompt: str, category: str) -> Generator[Dict[str, Any], None, None]:
    """vLLM 서버에 POST 요청하여 free-text 챌린지 응답을 SSE 형식으로 반환"""
    logger.info(f"[vLLM 호출] 프롬프트 길이: {len(prompt)}")
    url = "http://localhost:8800/v1/chat/completions"
    payload = {
        "model": "/home/ubuntu/mistral_finetuned_v5/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/0d4b76e1efeb5eb6f6b5e757c79870472e04bd3a",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": 512,
        "temperature": 0.7,
        "do_sample": True # temperature 설정 시 반드시 True로 설정해야 함: 확률적 샘플링 활성적
    }

    response_completed = False  # 응답 완료 여부를 추적하는 플래그
    token_buffer = ""  # 토큰을 누적할 버퍼
    # 누적 문장 버퍼 (추천문장 등 문장 단위 줄바꿈 플러시용)
    full_sentence = ""
    prev_token = None  # 이전 토큰 (숫자+점 조합 처리용)
    prev_prev_token = None  # 이전 이전 토큰 (숫자+점+공백 조합 처리용)
    # 한글과 영어 모두를 고려한 단어 구분자 (줄바꿈 포함)
    word_delimiters = [' ', '\t', '\n', '.', ',', '!', '?', ';', ':', '"', "'", '(', ')', '[', ']', '{', '}', '<', '>', '/','~', '은', '는', '가', '을', '를', '의', '에서', '으로', '와', '과', '만', '부터', '까지','든지', '라도', '으로', '께서', '분들께', '고', '며', '면', '거나', '든가', '위해', '위한', '도시에서', '바닷가에서', '산에서', '농촌에서', '사무직', '현장직', '영업직', '재택근무']

    try:
        with httpx.stream("POST", url, json=payload, timeout=60.0) as response:
            full_response = ""
            for line in response.iter_lines():
                if isinstance(line, bytes):
                    if line.startswith(b"data: "):
                        try:
                            json_data = json.loads(line[len(b"data: "):])
                            delta = json_data["choices"][0]["delta"]
                            token = delta.get("content", "")
                            if token.strip() in ["```", "`", ""]:
                                continue  # 이런 토큰은 누적하지 않음
                            full_response += token
                            token_buffer += token
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

                                        # 숫자+점+공백 조합 처리
                                        if (
                                            prev_prev_token is not None and prev_prev_token.isdigit() and
                                            prev_token is not None and re.match(r"^\.\s*$", prev_token) and
                                            word == " "
                                        ):
                                            yield {
                                                "event": "challenge",
                                                "data": json.dumps({
                                                    "status": 200,
                                                    "message": "토큰 생성",
                                                    "data": prev_prev_token + ". "
                                                }, ensure_ascii=False)
                                            }
                                            prev_token = None
                                            prev_prev_token = None
                                            continue

                                        # 숫자+점 조합 (공백 없음)
                                        if prev_token is not None and prev_token.isdigit() and re.match(r"^\.\s*$", word):
                                            yield {
                                                "event": "challenge",
                                                "data": json.dumps({
                                                    "status": 200,
                                                    "message": "토큰 생성",
                                                    "data": prev_token + "."
                                                }, ensure_ascii=False)
                                            }
                                            prev_token = None
                                            continue

                                        # 점만 있는 토큰 (선택적 공백/줄바꿈 포함) 무시
                                        if re.match(r"^\.\s*$", word):
                                            prev_prev_token = prev_token
                                            prev_token = None
                                            continue

                                        # 숫자만 있는 토큰 버퍼링
                                        if word.isdigit():
                                            prev_prev_token = prev_token
                                            prev_token = word
                                            continue

                                        # 다른 단어들에 대한 버퍼 시프트
                                        prev_prev_token = prev_token
                                        prev_token = word

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

                                        # 숫자+점+공백 조합 처리
                                        if (
                                            prev_prev_token is not None and prev_prev_token.isdigit() and
                                            prev_token is not None and re.match(r"^\.\s*$", prev_token) and
                                            word == " "
                                        ):
                                            yield {
                                                "event": "challenge",
                                                "data": json.dumps({
                                                    "status": 200,
                                                    "message": "토큰 생성",
                                                    "data": prev_prev_token + ". "
                                                }, ensure_ascii=False)
                                            }
                                            prev_token = None
                                            prev_prev_token = None
                                            continue

                                        # 숫자+점 조합 (공백 없음)
                                        if prev_token is not None and prev_token.isdigit() and re.match(r"^\.\s*$", word):
                                            yield {
                                                "event": "challenge",
                                                "data": json.dumps({
                                                    "status": 200,
                                                    "message": "토큰 생성",
                                                    "data": prev_token + "."
                                                }, ensure_ascii=False)
                                            }
                                            prev_token = None
                                            continue

                                        # 점만 있는 토큰 (선택적 공백/줄바꿈 포함) 무시
                                        if re.match(r"^\.\s*$", word):
                                            prev_prev_token = prev_token
                                            prev_token = None
                                            continue

                                        # 숫자만 있는 토큰 버퍼링
                                        if word.isdigit():
                                            prev_prev_token = prev_token
                                            prev_token = word
                                            continue

                                        # 다른 단어들에 대한 버퍼 시프트
                                        prev_prev_token = prev_token
                                        prev_token = word

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
            json_str = json_str.replace("```json", "").replace("```", "").replace("`", "").strip()
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
            json_str = re.sub(r'[ \t\r\f\v]+', ' ', json_str)
            logger.info(f"파싱 시도 문자열: {json_str}")
            
            # JSON 파싱 시도
            try:
                parsed_temp = json.loads(json_str)
                # rag_parser.parse() 안전하게 처리
                try:
                    parsed_data = rag_parser.parse(json.dumps(parsed_temp))
                except Exception as parse_error:
                    logger.error(f"rag_parser.parse() 실패: {str(parse_error)}")
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
                
                # 카테고리 정보 추가
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

# 대화 상태를 관리하기 위한 타입 정의
class ChatState(TypedDict):
    messages: Annotated[Sequence[str], "대화 기록"]
    current_query: str     # 사용자가 입력한 현재 질문
    context: str           # RAG 검색 자료
    response: str          # LLM 최종응답 
    should_continue: bool  # 대화 계속 여부
    error: Optional[str]   # 오류 메시지
    docs: Optional[list]   # 검색된 문서
    sessionId: str         # 세션 ID
    category: Optional[str]  # 현재 선택된 카테고리
    base_category: Optional[str]  # 원본 카테고리도 저장

def parse_challenges_string(challenges_str: str) -> list:
    """challenges 문자열을 파싱하여 리스트로 변환"""
    # 이미 리스트인 경우 그대로 반환
    if isinstance(challenges_str, list):
        return challenges_str
    
    # JSON 파싱 시도
    try:
        return json.loads(challenges_str)
    except:
        pass
    
    # 문자열이 아닌 경우 빈 리스트 반환
    if not isinstance(challenges_str, str):
        return []
    
    challenges = []
    current_challenge = {}
    
    # 줄 단위로 분리
    lines = challenges_str.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # 빈 줄 무시
        if not line:
            continue
            
        # 새로운 챌린지 시작
        if line.startswith('-') or line.startswith('*'):
            # 이전 챌린지가 있으면 추가
            if current_challenge and 'title' in current_challenge:
                challenges.append(current_challenge)
                current_challenge = {}
            
            # title 추출
            title_match = re.search(r'(?:title|제목)[\s:]*[\'"]?([^\'"]+)[\'"]?', line, re.IGNORECASE)
            if title_match:
                current_challenge['title'] = title_match.group(1).strip()
        
        # description 추출
        elif 'description' in line.lower() or '설명' in line:
            desc_match = re.search(r'(?:description|설명)[\s:]*[\'"]?([^\'"]+)[\'"]?', line, re.IGNORECASE)
            if desc_match:
                current_challenge['description'] = desc_match.group(1).strip()
    
    # 마지막 챌린지 추가
    if current_challenge and 'title' in current_challenge:
        challenges.append(current_challenge)
    
    return challenges

def format_sse_response(event: str, data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "event": event,
        "data": json.dumps(data, ensure_ascii=False)
    }

# 대화 그래프 노드 정의
def validate_query(state: ChatState) -> ChatState: # state는 챗봇의 현재 대화 상태를 담고 있는 딕셔너리
    """사용자 질문 유효성 검사"""
    if len(state["current_query"].strip()) < 5:
        state["error"] = "질문은 최소 5자 이상이어야 합니다."
        state["should_continue"] = False
    else:
        state["should_continue"] = True
    return state

def retrieve_context(state: ChatState) -> ChatState:
    """관련 컨텍스트 검색(RAG)"""
    if not state["should_continue"]:
        return state # 다음 단계로 진행할지를 결정하는 체크포인트 역할
    try:
        # RAG 검색 수행 (카테고리 필터 제거)
        docs = retriever.get_relevant_documents(state["current_query"])
        state["docs"] = docs
        state["context"] = "\n".join([doc.page_content for doc in docs])

        # 참조된 문서 로그 출력
        for idx, doc in enumerate(docs):
            print(f"[RAG 참조 문서 {idx+1}]")
            print(f"내용: {doc.page_content[:200]}")  # 너무 길면 일부만 출력
            print(f"메타데이터: {doc.metadata}")
        
        # 검색된 문서가 없는 경우
        if not docs:
            state["error"] = "관련된 챌린지 정보를 찾을 수 없습니다."
            state["should_continue"] = False
            
    except Exception as e:
        state["error"] = f"컨텍스트 검색 중 오류 발생: {str(e)}"
        state["should_continue"] = False
    
    return state


def generate_response(state: ChatState) -> ChatState:
    """응답 생성"""
    if not state["should_continue"]:
        return state
    try:
        messages = "\n".join(state["messages"])
        print(f"Generating response for query: {state['current_query']}")
        print(f"Current category in state: {state['category']}")
        
        category = state["category"]
        if category not in label_mapping:
            raise ValueError(f"잘못된 카테고리 값: {category}")
        eng_label = label_mapping[category]
        logger.info(f"Adding category info - eng: {eng_label}")
        
        # 프롬프트 생성
        prompt = custom_prompt.format(
            context=state["context"],
            query=state["current_query"],
            messages=messages,
            category=category
        )
        
        # LLM 응답 생성 (스트리밍 방식 유지)
        full_response = ""
        for data_payload in get_llm_response(prompt, category):
            if isinstance(data_payload, dict) and "data" in data_payload:
                full_response += str(data_payload["data"])
            yield data_payload

        if state["should_continue"]:
            print(f"Raw LLM response: {full_response}")
            
            # JSON 파싱 시도
            try:
                response_text = full_response
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1]
                if "```" in response_text:
                    response_text = response_text.split("```")[0]
                response_text = response_text.strip()
                
                # JSON 파싱
                parsed_response = json.loads(response_text)
                print(f"Successfully parsed JSON response. Length: {len(response_text)}")
                
                # 필수 필드 검증
                if "recommend" not in parsed_response or "challenges" not in parsed_response:
                    raise ValueError("응답에 필수 필드가 없습니다.")
                
                # challenges가 문자열인 경우 배열로 변환
                if isinstance(parsed_response.get("challenges"), str):
                    challenges = parse_challenges_string(parsed_response["challenges"])
                    parsed_response["challenges"] = challenges
                
                # challenges가 리스트가 아닌 경우 처리
                if not isinstance(parsed_response.get("challenges"), list):
                    raise ValueError("challenges는 리스트 형태여야 합니다.")
                
                # 현재 카테고리 정보로 챌린지 데이터 업데이트
                for challenge in parsed_response["challenges"]:
                    challenge["category"] = eng_label
                
                state["response"] = json.dumps(parsed_response, ensure_ascii=False)
                print(f"Final response with category: {category}, eng: {eng_label}")
                
            except ValueError as e:
                print(f"응답 검증 오류: {str(e)}")
                state["error"] = str(e)
                state["should_continue"] = False
                return state
        else:
            print(f"응답 검증 오류: {state['error']}")
            state["response"] = json.dumps({
                "recommend": "죄송합니다. 요청을 처리하는 중에 오류가 발생했습니다.",
                "challenges": []
            }, ensure_ascii=False)
        
        # 대화 기록 업데이트 (비교 코드 방식으로 개선)
        state["messages"] = list(state["messages"]) + [
            f"User: {state['current_query']}",
            f"Assistant: {state['response']}"
        ]
        
        return state
        
    except Exception as e:
        print(f"응답 생성 중 오류 발생: {str(e)}")
        state["error"] = f"응답 생성 중 오류 발생: {str(e)}"
        state["should_continue"] = False
        return state

def handle_error(state: ChatState) -> ChatState:
    """오류 처리"""
    if state["error"]:
        state["response"] = json.dumps({
            "recommend": "죄송합니다. 요청을 처리하는 중에 오류가 발생했습니다.",
            "challenges": []
        }, ensure_ascii=False)
    return state

def create_chat_graph():
    """대화 그래프 생성"""
    workflow = StateGraph(ChatState)
    
    # 노드 추가
    workflow.add_node("validate_query", validate_query)
    workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("generate_response", generate_response)
    workflow.add_node("handle_error", handle_error)
    
    # 엣지 추가
    workflow.add_edge("validate_query", "retrieve_context")
    workflow.add_edge("retrieve_context", "generate_response")
    workflow.add_edge("generate_response", END)
    workflow.add_edge("handle_error", END)
    
    # 조건부 엣지 추가
    workflow.add_conditional_edges(
        "validate_query",
        lambda x: "retrieve_context" if x["should_continue"] else "handle_error"
    )
    
    workflow.add_conditional_edges(
        "retrieve_context",
        lambda x: "generate_response" if x["should_continue"] else "handle_error"
    )
    
    workflow.add_conditional_edges(
        "generate_response",
        lambda x: END if x["should_continue"] else "handle_error"
    )
    
    # 시작 노드 설정
    workflow.set_entry_point("validate_query")
    
    return workflow.compile()

# 대화 그래프 인스턴스 생성
chat_graph = create_chat_graph()

# 대화 상태 저장소
conversation_states = {}

def process_chat(sessionId: str, query: str, base_info_category: Optional[str] = None) -> str:
    """대화 처리 함수"""
    print(f"\n🚀🚀🚀 FREE-TEXT PROCESS CHAT START 🚀🚀🚀")
    print(f"🔥 Initial base_info_category: {base_info_category}")
    print(f"🔥 User query: {query}")
    print(f"🔥 Session ID: {sessionId}")
    print(f"🔥 코드 버전: 2024-07-27-v2")

    # 이전 대화 상태 가져오기 또는 새로 생성
    if sessionId not in conversation_states:
        # free-text에서 base_info_category가 없으면 기본값 설정
        if not base_info_category:
            base_info_category = "제로웨이스트"  # 기본 카테고리
            print(f"🔥 No base_info_category provided. Using default: {base_info_category}")
        if base_info_category not in label_mapping:
            raise ValueError(f"잘못된 카테고리 값: {base_info_category}")
            
        print(f"🔥 New session detected. Initializing with category: {base_info_category}")
        conversation_states[sessionId] = {
            "messages": [],             # 대화 기록 
            "current_query": "",        # 사용자가 입력한 현재 질문
            "context": "",              # RAG 검색 자료
            "response": "",             # LLM 최종응답 
            "should_continue": True,    # 대화 진행 가능성 여부
            "error": None,
            "docs": None,               # 검색된 원본 문서 리스트 (Qdrant의 Document 객체들)
            "sessionId": sessionId,
            "category": base_info_category,  # base-info 카테고리 저장 -> 사용자에 요청에 따라 변경되는 카테고리
            "base_category": base_info_category  # 원본 카테고리도 저장
        }
        # 초기 카테고리 설정 로그
        conversation_states[sessionId]["messages"].append(f"Initial category set to {base_info_category}")
    else:
        print(f"현재 카테고리: {conversation_states[sessionId]['category']}")
    
    # 현재 상태 업데이트
    state = conversation_states[sessionId]
    state["current_query"] = query
    print(f"Current state category before random: {state['category']}")

    # 카테고리 변경 처리
    category_changed = False

    # 1. "원래 카테고리로" 요청 처리
    if any(keyword in query.lower() for keyword in ["원래", "처음", "이전", "원래대로","기존"]):
        if state["base_category"]:
            state["category"] = state["base_category"]
            state["messages"].append(f"Category restored to original: {state['base_category']}")
            category_changed = True

    # 2. "아무거나" 등의 요청 처리
    elif any(keyword in query.lower() for keyword in ["아무", "아무거나", "다른거", "새로운거", "딴거", "다른"]):
        available_categories = [cat for cat in label_mapping.keys() if cat != state["category"]]
        if not available_categories:
            available_categories = list(label_mapping.keys())
        
        sampled_category = random.choice(available_categories)
        state["category"] = sampled_category
        state["messages"].append(f"Category randomly selected: {sampled_category}")
        category_changed = True

    # 3. 특정 카테고리 요청 처리 (키워드 기반)
    else:
        query_lower = query.lower()
        print(f"🔍 키워드 검색 시작: '{query_lower}'")
        for category, keywords in category_keywords.items():
            print(f"   - 카테고리 '{category}' 키워드 확인: {keywords}")
            if any(keyword in query_lower for keyword in keywords):
                print(f"키워드 매칭 성공! '{category}' 카테고리로 변경")
                state["category"] = category
                state["messages"].append(f"Category changed to {category} based on keywords")
                category_changed = True
                break
        if not category_changed:
            print(f"매칭되는 키워드 없음. 기존 카테고리 유지: {state['category']}")

    # 4. base-info 카테고리 처리
    if not category_changed and base_info_category and state["category"] != base_info_category:
        state["category"] = base_info_category
        state["messages"].append(f"Category changed to {base_info_category}")
        category_changed = True

    print(f"State category before chat_graph: {state['category']}")

    # 대화 그래프 실행
    result = chat_graph.invoke(state)
    
    # 응답 생성 시 현재 카테고리 정보 포함
    try:
        if not result["response"]:
            raise ValueError("응답이 비어있습니다.")
            
        response_data = json.loads(result["response"])
        current_category = result["category"]
        print(f"Current category in result: {current_category}")
        
        if current_category not in label_mapping:
            raise ValueError(f"잘못된 카테고리 값: {current_category}")
            
        eng_label = label_mapping[current_category]
        
        # 챌린지 데이터에 현재 카테고리 정보 업데이트
        if "challenges" in response_data:
            for challenge in response_data["challenges"]:
                challenge["category"] = eng_label
        
        # 업데이트된 응답으로 result 수정
        result["response"] = json.dumps(response_data, ensure_ascii=False)
        
        # 상태 저장
        conversation_states[sessionId] = result
        print(f"Final state category: {result['category']}")
        
        return result["response"]
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 오류: {str(e)}")
        print(f"Raw response: {result.get('response', '')}")
        # 상태 저장
        conversation_states[sessionId] = result
        return json.dumps({
            "recommend": "죄송합니다. 응답을 처리하는 중에 오류가 발생했습니다.",
            "challenges": []
        }, ensure_ascii=False)
    except Exception as e:
        print(f"Error in response processing: {str(e)}")
        # 상태 저장
        conversation_states[sessionId] = result
        return json.dumps({
            "recommend": "죄송합니다. 응답을 처리하는 중에 오류가 발생했습니다.",
            "challenges": []
        }, ensure_ascii=False)
    finally:
        # 메모리 정리
        # shared_model.cleanup_memory()
        logger.info("process_chat 메모리 정리 완료")

def clear_conversation(sessionId: str):
    """대화 기록 삭제"""
    if sessionId in conversation_states:
        del conversation_states[sessionId]

def get_conversation_history(sessionId: str) -> List[str]:
    """대화 기록 조회
    Args:
        sessionId: 사용자 세션 ID
    
    Returns:
        List[str]: 대화 기록 리스트
    """
    if sessionId in conversation_states:
        return conversation_states[sessionId]["messages"]
    return []
