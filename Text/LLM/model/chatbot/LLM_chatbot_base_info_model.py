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

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

# 프로젝트 루트 경로 설정
current_file = os.path.abspath(__file__)
logger.info(f"Current file: {current_file}")

project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
logger.info(f"Project root: {project_root}")

# 모델 경로 설정
MODEL_PATH = os.path.join(project_root, "mistral")
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
        torch_dtype=torch.float16
    )
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        cache_dir=MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
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
    ResponseSchema(name="recommend", description="추천 텍스트를 한 문장으로 출력해줘.(예: '이런 챌린지를 추천합니다.')"),
    ResponseSchema(name="challenges", description="추천 챌린지 리스트, 각 항목은 title, description 포함, description은 한 문장으로 요약해주세요.")
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

def get_llm_response(prompt: str) -> Generator[str, None, None]:
    """LLM 응답을 SSE 형식으로 반환 (서버에서 전체 파싱 후 전달)"""
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
            inputs,
            streamer=streamer,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        # 전체 응답 누적용
        full_response = ""

        for new_text in streamer:
            if new_text:
                full_response += new_text
                yield format_sse_response("challenge", {
                    "status": 200,
                    "message": "토큰 생성",
                    "data": {"token": new_text}
                })

        # 서버에서 직접 파싱
        try:
            parsed_data = base_parser.parse(full_response.strip())
        except Exception as e:
            yield format_sse_response("error", {
                    "status": 500,
                    "message": f"파싱 실패: {str(e)}",
                    "data": None
            })
            return

        # 종료 이벤트
        yield format_sse_response("close", {
            "status": 200,
            "message": "모든 응답 완료",
            "data": parsed_data
        })

    except Exception as e:
        yield format_sse_response("error", {
                "status": 500,
            "message": f"예외 발생: {str(e)}",
                "data": None
        })
