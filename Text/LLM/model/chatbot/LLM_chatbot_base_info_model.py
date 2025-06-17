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
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
from transformers import LogitsProcessorList, InfNanRemoveLogitsProcessor
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
import gc


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
        "mistralai/Mistral-7B-Instruct-v0.3",  # Mistral-7B 모델의 토크나이저 로드
        cache_dir=MODEL_PATH,  # 모델 파일을 저장할 로컬 경로
        torch_dtype=torch.float16,  # 16비트 부동소수점 사용으로 메모리 사용량 절반으로 감소
        token=hf_token  # Hugging Face API 토큰으로 비공개 모델 접근
    )
    # 패딩 토큰 설정 - Mistral 모델은 기본 패딩 토큰이 없어서 EOS 토큰으로 대체
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    logger.info("Loading model...")
    # 4비트 양자화 설정 - 모델 크기를 75% 감소시키면서 성능 유지
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 4비트 양자화 활성화
        bnb_4bit_compute_dtype=torch.float16,  # 계산은 16비트로 수행
        bnb_4bit_use_double_quant=True,  # 이중 양자화로 메모리 추가 절약
        bnb_4bit_quant_type="nf4"  # (Normalized Float 4‑bit)
    )

    # GPU 메모리 사용량 계산
    gpu_memory = torch.cuda.get_device_properties(0).total_memory
    # 4비트 양자화된 Mistral-7B 모델은 약 4GB의 메모리를 사용
    model_memory = 4 * 1024**3  # 4GB
    # 남은 메모리의 90%를 사용 가능하도록 설정
    available_memory = int((gpu_memory - model_memory) * 0.9)
    logger.info(f"GPU 메모리: {gpu_memory / 1024**3:.2f}GB, 모델 예상 메모리: {model_memory / 1024**3:.2f}GB, 사용 가능 메모리: {available_memory / 1024**3:.2f}GB")
    
    # 모델 로드 전 메모리 정리
    torch.cuda.empty_cache()
    gc.collect()
    
    # 모델 로드 시 메모리 최적화 옵션 추가
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",  # Mistral-7B 모델 로드
        cache_dir=MODEL_PATH,  # 모델 파일을 저장할 로컬 경로
        device_map="auto",  # GPU/CPU 자동 할당
        low_cpu_mem_usage=True,  # CPU 메모리 사용량 최소화
        token=hf_token,  # Hugging Face API 토큰
        torch_dtype=torch.float16,  # 16비트 부동소수점 사용
        trust_remote_code=True,  # 커스텀 코드 실행 허용
        max_position_embeddings=2048,  # 최대 입력 길이 제한
        quantization_config=quantization_config,  # 4비트 양자화 적용
        offload_folder="offload",  # 메모리 부족시 모델 일부를 디스크에 저장
        offload_state_dict=True  # 모델 상태를 디스크에 저장하여 메모리 절약
    )
    
    # 메모리 최적화를 위한 설정
    model.config.use_cache = False  # 캐시 사용 비활성화
    model.eval()
    
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

logger.info("Model loaded successfully!")

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
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("메모리 정리 완료")
        except Exception as e:
            logger.error(f"메모리 정리 중 에러 발생: {str(e)}")
