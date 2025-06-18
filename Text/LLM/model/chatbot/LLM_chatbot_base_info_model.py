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
from Text.LLM.model.chatbot.shared_model import shared_model

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

# 공유 모델 사용
model = shared_model.model
tokenizer = shared_model.tokenizer

logger.info("Using shared Mistral model for base-info chatbot")

# base-info_response_schemas 정의
base_response_schemas = [
    ResponseSchema(name="recommend", description="추천 텍스트를 한글로 한 문장으로 출력해줘.(예: '이런 챌린지를 추천합니다.')"),
    ResponseSchema(name="challenges", description="추천 챌린지 리스트, 각 항목은 title, description 포함, description은 한글로 한 문장으로 요약해주세요.")
]

# base-info_output_parser 정의 
base_parser = StructuredOutputParser.from_response_schemas(base_response_schemas)

# base-info_prompt 정의
escaped_format = base_parser.get_format_instructions().replace("{", "{{").replace("}", "}}")
base_prompt = PromptTemplate(
    input_variables=["location", "workType", "category"],
    template=f"""<s>[INST] 당신은 환경 보호 챌린지를 추천하는 AI 어시스턴트입니다.
{{location}}의 환경에 있는 {{workType}} 사용자가 {{category}}를 실천할 때,
절대적으로 환경에 도움이 되는 챌린지를 아래 JSON 형식으로 3가지 추천해주세요.

주의사항:
1. 모든 속성 이름은 반드시 큰따옴표(")로 둘러싸야 합니다.
2. 모든 문자열 값도 큰따옴표(")로 둘러싸야 합니다.
3. recommend 필드에는 {{category}} 관련 추천 문구를 포함해야 합니다.

JSON 포맷:
{escaped_format}

반드시 위 JSON 형식 그대로 반드시 한글로 한번만 출력하세요. [/INST]</s>
"""
)

def format_sse_response(event: str, data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "event": event,
        "data": json.dumps(data, ensure_ascii=False)
    }

def get_llm_response(prompt: str, category: str) -> Generator[Dict[str, Any], None, None]:
    """LLM 응답을 SSE 형식으로 반환 (서버에서 전체 파싱 후 전달)"""
    logger.info(f"LLM 응답 생성 시작 - 프롬프트 길이: {len(prompt)}")
    try:
        # 메모리 정리
        shared_model.cleanup_memory()
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        logger.info(f"토크나이저 입력 준비 완료. 입력 토큰 수: {inputs.input_ids.shape[1]}")
        
        # 스트리머 설정
        streamer = TextIteratorStreamer(
            tokenizer, # 토큰을 텍스트로 변환하는 데 사용할 토크나이저
            skip_prompt=True, # skip_prompt: True로 설정하면 입력 프롬프트는 스트리밍에서 제외하고 모델의 응답만 스트리밍
            timeout=None,  # timeout: None으로 설정하여 무한정 대기 (응답이 늦어도 연결 유지)
            decode_kwargs={ # decode_kwargs: 토큰을 텍스트로 변환할 때 사용할 추가 옵션
                "skip_special_tokens": True, # True로 설정하여 [PAD], [CLS] 등의 특수 토큰을 제외
                "clean_up_tokenization_spaces": True # 토큰화 공백 정리
            }
        )

        # inf, nan 값 처리를 위한 로짓 프로세서 설정
        logits_processor = LogitsProcessorList([
            InfNanRemoveLogitsProcessor()
        ])

        # 모델 생성 설정
        generation_kwargs = dict(
            inputs,  # 입력 텐서 (input_ids, attention_mask 등)
            streamer=streamer,  # 스트리밍 응답을 위한 TextIteratorStreamer 객체
            max_new_tokens=1024,  # 모델이 생성할 수 있는 최대 토큰 수 (JSON이 완성되도록 충분히 설정)
            temperature=0.7,  # 생성 다양성 조절 (0.0~1.0, 높을수록 더 다양한 응답)
            do_sample=True,  # 확률적 샘플링 활성화 (temperature와 함께 사용)
            pad_token_id=tokenizer.eos_token_id, # 패딩 토큰 ID 설정 (Mistral은 EOS 토큰을 패딩으로 사용)
            logits_processor=logits_processor  
        )
        logger.info("스레드 시작 및 모델 생성 시작.")
        
        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        # 전체 응답 누적용
        full_response = ""
        logger.info("스트리밍 응답 대기 중...")
        response_completed = False  # 응답 완료 여부를 추적하는 플래그

        try:
            # 스트리밍 응답 처리
            for new_text in streamer:
                if new_text and not response_completed:  # 응답이 완료되지 않은 경우에만 처리
                    full_response += new_text
                    logger.info(f"토큰 수신: {new_text[:20]}...")
                    
                    # 토큰 정제 - 순수 텍스트만 추출
                    cleaned_text = new_text
                    # JSON 관련 문자열 제거
                    cleaned_text = re.sub(r'"(recommend|challenges|title|description)":\s*("|\')?', '', cleaned_text)
                    # 마크다운 및 JSON 구조 제거
                    cleaned_text = cleaned_text.replace("```json", "").replace("```", "").strip()
                    cleaned_text = re.sub(r'["\']', '', cleaned_text)  # 따옴표 제거
                    cleaned_text = re.sub(r'[\[\]{}]', '', cleaned_text)  # 괄호 제거
                    cleaned_text = re.sub(r',\s*$', '', cleaned_text)  # 끝의 쉼표 제거
                    # 불필요한 공백 제거
                    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
                    cleaned_text = cleaned_text.strip()
                    
                    if cleaned_text:
                        # SSE 응답 전송 - 순수 텍스트만 전송
                        yield {
                            "event": "challenge",
                            "data": json.dumps({
                                "status": 200,
                                "message": "토큰 생성",
                                "data": cleaned_text
                            }, ensure_ascii=False)
                        }

            # 스레드 완료 대기
            thread.join()
            
            # 토큰 캐시 정리
            if hasattr(streamer, 'token_cache'):
                streamer.token_cache = []
            
            # 전체 응답 파싱
            logger.info("스트리밍 완료. 전체 응답 파싱 시작.")
            try:
                # 전체 응답 로깅
                logger.info(f"전체 응답: {full_response}")
                
                # JSON 문자열 추출
                json_match = re.search(r"```json\n([\s\S]*?)\n```", full_response.strip())
                if json_match:
                    json_string_to_parse = json_match.group(1).strip()
                    logger.info(f"JSON 추출: {json_string_to_parse}")
                else:
                    json_string_to_parse = full_response.strip()
                    logger.info(f"JSON 추출 실패, 전체 응답 사용: {json_string_to_parse}")

                if not json_string_to_parse.strip():
                    raise ValueError("JSON 문자열이 비어있습니다")

                # JSON 파싱 전 문자열 정제
                # 객체와 배열의 마지막 쉼표 제거
                json_string_to_parse = re.sub(r',(\s*[}\]])', r'\1', json_string_to_parse)
                # 불필요한 공백 제거
                json_string_to_parse = re.sub(r'\s+', ' ', json_string_to_parse)
                # 연속된 쉼표 제거
                json_string_to_parse = re.sub(r',\s*,', ',', json_string_to_parse)
                
                # logger.info(f"정제된 JSON 문자열: {json_string_to_parse}")

                # JSON 파싱
                try:
                    parsed_data_temp = json.loads(json_string_to_parse)
                    parsed_data = base_parser.parse(json.dumps(parsed_data_temp))
                    logger.info(f"파싱 성공: {parsed_data}")
                except json.JSONDecodeError as e:
                    logger.error(f"JSON 파싱 실패: {str(e)}")
                    logger.error(f"파싱 시도한 문자열: {json_string_to_parse}")
                    response_completed = True
                    raise

                # 카테고리 정보 추가
                eng_label = label_mapping[category]
                if isinstance(parsed_data, dict) and "challenges" in parsed_data:
                    for challenge in parsed_data["challenges"]:
                        challenge["category"] = eng_label
                    logger.info(f"카테고리 추가 완료: {eng_label}")
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
                else:
                    raise ValueError("파싱된 데이터에 'challenges' 필드가 없습니다.")

            except Exception as e:
                logger.error(f"파싱 실패: {str(e)}")
                response_completed = True  # 에러 발생 시에도 응답 완료 플래그 설정
                yield {
                    "event": "error",
                    "data": json.dumps({
                        "status": 500,
                        "message": f"파싱 실패: {str(e)}",
                        "data": None
                    }, ensure_ascii=False)
                }

        except Exception as e:
            logger.error(f"스트리밍 중 에러 발생: {str(e)}")
            response_completed = True  # 에러 발생 시에도 응답 완료 플래그 설정
            yield {
                "event": "error",
                "data": json.dumps({
                    "status": 500,
                    "message": f"스트리밍 에러: {str(e)}",
                    "data": None
                }, ensure_ascii=False)
            }

    except Exception as e:
        logger.error(f"=== 예외 발생 ===")
        logger.error(f"예외 타입: {type(e).__name__}")
        logger.error(f"예외 메시지: {str(e)}")
        yield format_sse_response("error", {
            "status": 500,
            "message": f"예외 발생: {str(e)}",
            "data": None
        })

    finally:
        # 요청 완료 후 메모리 정리
        try:
            if 'inputs' in locals():
                del inputs
            shared_model.cleanup_memory()
            logger.info("메모리 정리 완료")
        except Exception as e:
            logger.error(f"메모리 정리 중 에러 발생: {str(e)}")
