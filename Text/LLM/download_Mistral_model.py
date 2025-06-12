from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from dotenv import load_dotenv
import logging
import subprocess
import sys
import platform
from huggingface_hub import login

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_python_version():
    """
    Python 버전을 확인하고 호환성을 체크합니다.
    """
    python_version = platform.python_version_tuple()
    major, minor = int(python_version[0]), int(python_version[1])
    
    if major > 3 or (major == 3 and minor > 11):
        logger.warning(f"현재 Python 버전 {platform.python_version()}은(는) 일부 라이브러리와 호환성 문제가 있을 수 있습니다.")
        return False
    return True

def check_dependencies():
    """
    필요한 의존성 패키지들을 확인하고 설치합니다.
    """
    required_packages = {
        'sentencepiece': 'sentencepiece',
        'transformers': 'transformers',
        'torch': 'torch',
        'python-dotenv': 'python-dotenv'
    }
    
    missing_packages = []
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(pip_name)
    
    if missing_packages:
        logger.info(f"필요한 패키지 설치 중: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            logger.info("패키지 설치 완료")
        except subprocess.CalledProcessError as e:
            logger.error(f"패키지 설치 실패: {str(e)}")
            return False
    return True

# 환경 변수 로드
load_dotenv()

# Hugging Face 로그인
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEYMAC")
if HF_API_KEY:
    login(HF_API_KEY)
    logger.info("Hugging Face 로그인 성공")
else:
    logger.warning("Hugging Face API 키가 설정되지 않았습니다.")

def download_model():
    """
    Mistral 모델과 토크나이저를 다운로드하고 로컬에 저장합니다.
    """
    try:
        # Python 버전 체크
        if not check_python_version():
            user_input = input("계속 진행하시겠습니까? (y/n): ")
            if user_input.lower() != 'y':
                return False

        # 의존성 체크
        if not check_dependencies():
            return False

        # 모델 이름 설정
        model_name = os.getenv("HF_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
        local_dir = "./mistral"  # 로컬 저장 경로를 ./mistral로 변경

        # 저장 디렉토리 생성
        os.makedirs(local_dir, exist_ok=True)

        logger.info(f"모델 다운로드 시작: {model_name}")
        
        # 토크나이저 다운로드
        logger.info("토크나이저 다운로드 중...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=local_dir,
            trust_remote_code=True,
            use_fast=True,
            token=HF_API_KEY
        )
        logger.info("토크나이저 다운로드 완료")

        # 모델 다운로드
        logger.info("모델 다운로드 중...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=local_dir,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            token=HF_API_KEY
        )
        logger.info("모델 다운로드 완료")

        logger.info(f"모델이 성공적으로 다운로드되어 {local_dir}에 저장되었습니다.")
        return True

    except Exception as e:
        logger.error(f"모델 다운로드 중 오류 발생: {str(e)}")
        return False

if __name__ == "__main__":
    success = download_model()
    if success:
        logger.info("모델 다운로드가 성공적으로 완료되었습니다.")
    else:
        logger.error("모델 다운로드에 실패했습니다.")
