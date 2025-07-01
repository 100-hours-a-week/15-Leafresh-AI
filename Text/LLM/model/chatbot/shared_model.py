from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os
import logging
from huggingface_hub import login
import gc
from fastapi import HTTPException
from dotenv import load_dotenv

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SharedMistralModel:
    _instance = None
    _model = None
    _tokenizer = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SharedMistralModel, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        print("Feedback Model 생성!")
        # 즉시 로딩: 인스턴스 생성 시 무조건 모델을 메모리에 올림
        if not self._initialized:
            print("Feedback Model 초기화!")
            self._initialize_model()
            self._initialized = True
    
    def _initialize_model(self):
        """모델 초기화 - 한 번만 실행됨"""
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

        # GPU 사용 가능 여부 확인
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"사용 가능한 디바이스: {device}")

        try:
            logger.info("Loading tokenizer...")
            self._tokenizer = AutoTokenizer.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.3",
                cache_dir=MODEL_PATH,
                torch_dtype=torch.float16,
                token=hf_token
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
                self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
            
            logger.info("Loading model...")

            # 4비트 양자화 설정 - 안정성과 효율성을 위해 4비트 유지
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,  # 4비트 양자화 활성화
                bnb_4bit_compute_dtype=torch.float16,  # 계산은 16비트로 수행
                bnb_4bit_use_double_quant=True,  # 이중 양자화로 메모리 추가 절약
                bnb_4bit_quant_type="nf4"  # Normalized Float 4-bit
            )

            # GPU 메모리 사용량 계산
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            model_memory = 4 * 1024**3  # 4GB (4비트 양자화)
            available_memory = int((gpu_memory - model_memory) * 0.9)
            logger.info(f"GPU 메모리: {gpu_memory / 1024**3:.2f}GB, 모델 예상 메모리: {model_memory / 1024**3:.2f}GB, 사용 가능 메모리: {available_memory / 1024**3:.2f}GB")

            # 모델 로드 전 메모리 정리
            torch.cuda.empty_cache()
            gc.collect()

            self._model = AutoModelForCausalLM.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.3",
                cache_dir=MODEL_PATH,
                device_map="auto",
                low_cpu_mem_usage=True,
                token=hf_token,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                max_position_embeddings=2048,
                quantization_config=quantization_config,
                offload_folder="offload",
                offload_state_dict=True,
            )
            
            # 메모리 최적화를 위한 설정
            self._model.config.use_cache = False
            self._model.eval()

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

        logger.info("Shared Mistral model loaded successfully!")
    
    @property
    def model(self):
        return self._model
    
    @property
    def tokenizer(self):
        return self._tokenizer
    
    def cleanup_memory(self):
        """메모리 정리"""
        try:
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("Shared model memory cleanup completed")
        except Exception as e:
            logger.error(f"Memory cleanup error: {str(e)}")

# 전역 인스턴스 생성 (모듈 로드 시점에 생성되지만 로그는 출력하지 않음)
shared_model = SharedMistralModel() 