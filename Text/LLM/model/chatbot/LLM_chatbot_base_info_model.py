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
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
import os
import json
import random
import re
import threading
import logging
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from huggingface_hub import login

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

# Hugging Face 로그인
hf_token = os.getenv("HUGGINGFACE_API_KEYMAC")
if hf_token:
    try:
        login(token=hf_token)
        logger.info("Hugging Face Hub에 성공적으로 로그인했습니다.")
    except Exception as e:
        logger.error(f"Hugging Face Hub 로그인 실패: {e}")
else:
    logger.warning("HUGGINGFACE_API_KEYMAC 환경 변수를 찾을 수 없습니다. Hugging Face 로그인 건너뜜.")

# 프로젝트 루트 경로 설정
current_file = os.path.abspath(__file__)
logger.info(f"Current file: {current_file}")

project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
logger.info(f"Project root: {project_root}")

# 모델 경로 설정
MODEL_PATH = "/home/ubuntu/mistral"
logger.info(f"Model path: {MODEL_PATH}")

# 모델 경로 확인
if not os.path.exists(MODEL_PATH):
    logger.error(f"모델 경로를 찾을 수 없습니다: {MODEL_PATH}")
    raise HTTPException(
        status_code=500,
        detail={
            "status": 500,
            "message": f"모델 경로를 찾을 수 없습니다: {MODEL_PATH}",
            "data": None
        }
    )

logger.info(f"모델 경로: {MODEL_PATH}")

# GPU 사용 가능 여부 확인
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"사용 가능한 디바이스: {device}")

# 모델과 토크나이저 초기화
try:
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        cache_dir=MODEL_PATH,
        torch_dtype=torch.float16,
        token=hf_token
    )
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        cache_dir=MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        token=hf_token
    )
except Exception as e:
    logger.error(f"모델 로딩 실패: {str(e)}")
    raise HTTPException(
        status_code=500,
        detail={
            "status": 500,
            "message": f"모델 로딩 실패: {str(e)}",
            "data": None
        }
    )

# CPU를 사용하는 경우 모델을 CPU로 이동
if device == "cpu":
    model = model.to(device)

logger.info("Model loaded successfully!")

# base-info_response_schemas 정의
base_response_schemas = [
    ResponseSchema(name="recommend", description="추천 텍스트를 한 문장으로 출력해줘."),
    ResponseSchema(name="challenges", description="추천 챌린지 리스트, 각 항목은 title, description 포함, title은 description 요약문으로 10자 이내로 작성")
]

# base-info_output_parser 정의 
base_parser = StructuredOutputParser.from_response_schemas(base_response_schemas)

# base-info_prompt 정의
escaped_format = base_parser.get_format_instructions().replace("{", "{{").replace("}", "}}")
base_prompt = PromptTemplate(
    input_variables=["location", "workType", "category"],
    template=f"""<s>[INST] 당신은 환경 보호 챌린지를 추천하는 AI 어시스턴트입니다.
{{location}} 환경에 있는 {{workType}} 사용자가 {{category}}를 실천할 때,
절대적으로 환경에 도움이 되는 챌린지를 아래 JSON 형식으로 3가지 추천해주세요.

주의사항:
1. 모든 속성 이름은 반드시 큰따옴표(")로 둘러싸야 합니다.
2. 모든 문자열 값도 큰따옴표(")로 둘러싸야 합니다.
3. recommend 필드에는 {{category}} 관련 추천 문구를 포함해야 합니다.

JSON 포맷:
{escaped_format}

응답은 반드시 한글로 위 JSON형식 그대로 출력하세요. [/INST]</s>
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
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        logger.info(f"토크나이저 입력 준비 완료. 입력 토큰 수: {inputs.input_ids.shape[1]}")
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
            inputs,
            streamer=streamer,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        logger.info("스레드 시작 및 모델 생성 시작.")
        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        # 전체 응답 누적용
        full_response = ""
        logger.info("스트리밍 응답 대기 중...")

        for new_text in streamer:
            if new_text:
                full_response += new_text
                logger.info(f"토큰 수신: {new_text[:20]}...") # 처음 20자만 로깅하여 너무 길어지지 않게 함
                yield {
                    "event": "challenge",
                    "data": json.dumps({
                        "status": 200,
                        "message": "토큰 생성",
                        "data": {"token": new_text}
                    }, ensure_ascii=False)
                }

        logger.info("스트리밍 완료. 전체 응답 파싱 시작.")
        # 서버에서 직접 파싱
        try:
            # 마크다운 코드 블록 제거
            json_match = re.search(r"```json\n([\s\S]*?)\n```", full_response.strip())
            if json_match:
                json_string_to_parse = json_match.group(1).strip()
            else:
                json_string_to_parse = full_response.strip()

            # JSON 문자열 클리닝: 비표준 따옴표 및 불필요한 문자 제거 시도
            json_string_to_parse = json_string_to_parse.replace(""", '"').replace(""", '"')
            json_string_to_parse = re.sub(r'\s*\)\s*,', ',', json_string_to_parse) # 잘못된 괄호 뒤 콤마 제거
            
            # JSON 블록 뒤에 붙는 불필요한 텍스트 제거 (최대한 JSON만 남기도록)
            # 마지막 } 괄호 뒤의 모든 것을 제거
            last_brace_index = json_string_to_parse.rfind('}')
            if last_brace_index != -1:
                json_string_to_parse = json_string_to_parse[:last_brace_index + 1]

            logger.info(f"파싱 전 JSON 문자열 (클린징 후): {json_string_to_parse[:500]}...") # 너무 길어지지 않게 처음 500자만 로깅

            # Langchain 파서를 사용하기 전에 json.loads로 먼저 파싱 시도
            # 이렇게 하면 더 일반적인 JSON 오류를 잡을 수 있습니다.
            parsed_data_temp = json.loads(json_string_to_parse)
            
            # Langchain 파서로 최종 파싱 (필요하다면)
            parsed_data = base_parser.parse(json.dumps(parsed_data_temp))

            # ADDED: 파싱된 데이터의 각 챌린지에 카테고리 정보 추가
            eng_label = label_mapping[category] # category는 외부에서 주입된 변수
            logger.info(f"Adding category info - eng: {eng_label}")
            if isinstance(parsed_data, dict) and "challenges" in parsed_data:
                for challenge in parsed_data["challenges"]:
                    challenge["category"] = eng_label
                    logger.info(f"Added category info to challenge: {challenge['title']}")

            logger.info("파싱 성공.")
        except Exception as e:
            logger.error(f"파싱 실패: {str(e)}")
            logger.error(f"문제가 된 전체 응답 (클린징 전): {full_response}")
            logger.error(f"문제가 된 JSON 문자열 (클린징 후): {json_string_to_parse}")
            yield format_sse_response("error", {
                    "status": 500,
                    "message": f"파싱 실패: {str(e)}",
                    "data": None
            })
            return

        # 종료 이벤트
        logger.info("모든 응답 완료 및 종료 이벤트 전송.")
        yield {
            "event": "close",
            "data": json.dumps({
                "status": 200,
                "message": "모든 응답 완료",
                "data": parsed_data
            }, ensure_ascii=False)
        }

    except Exception as e:
        logger.error(f"예외 발생: {str(e)}")
        yield {
            "event": "error",
            "data": json.dumps({
                "status": 500,
                "message": f"예외 발생: {str(e)}",
                "data": None
            }, ensure_ascii=False)
        }
