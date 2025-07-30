from typing import List, Dict, Any, AsyncIterator
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import traceback
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import json
import logging
from huggingface_hub import login
import gc
# from Text.LLM.model.chatbot.shared_model import shared_model
import requests

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

VLLM_API_URL = os.getenv("VLLM_API_URL", "http://localhost:8800/v1/chat/completions")

class VLLMFeedbackClient:
    def __init__(self):
        self.api_url = VLLM_API_URL

    def get_feedback(self, messages, temperature=0.7, max_tokens=512, model=...):
        """
        vLLM(OpenAI 호환) 서버에 채팅 메시지로 inference 요청
        messages: OpenAI API 포맷의 메시지 리스트
        """
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": "/home/ubuntu/mistral_finetuned_v5/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/0d4b76e1efeb5eb6f6b5e757c79870472e04bd3a",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            # OpenAI 호환 응답: data["choices"][0]["message"]["content"]
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"vLLM API 요청 실패: {e}")

# 전역 싱글톤 인스턴스
shared_model = VLLMFeedbackClient() 

class FeedbackModel:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FeedbackModel, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            load_dotenv()
            
            # 지연 로딩: 실제 사용할 때만 모델 로드
            self._model = None
            self._tokenizer = None
            
            # 한글 기준으로 5-6문장에 적절한 토큰 수로 조정 (약 250-300자)
            self.max_tokens = 2048
            # 프롬프트 템플릿을 환경 변수에서 가져오거나 기본값 사용
            self.prompt_template = os.getenv("FEEDBACK_PROMPT_TEMPLATE",
            """
            너는 피드백 어시스턴트야. 아래와 같은 JSON 입력을 받으면, 사용자의 챌린지 활동을 요약해서 칭찬과 격려를 한글로 해줘.
            1. 유니코드(Unicode) 표준에 포함된 이모지(예: 😊, 🌱, 🎉 등)를 적절히 사용하여 친근하고 밝은 톤으로 작성해주세요.
            2. 성공한 챌린지에 대해서는 사용자의 노력을 인정하고 격려하는 메세지를 포함해주고, 실패한 챌린지에 대해서는 위로와 함께 다음 기회를 기대한다는 메시지를 포함해주세요.
            3. {personal_challenges}와 {group_challenges} 기록을 통합하여 구체적으로 완결된 문장으로 평가해주세요.
            개인 챌린지:
            {personal_challenges}
            단체 챌린지:
            {group_challenges}
            """)
            
            self._initialized = True

    @property
    def model(self):
        if self._model is None:
            # self._model = shared_model.model
            pass
        return self._model
    
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            # self._tokenizer = shared_model.tokenizer
            pass
        return self._tokenizer

    def _is_within_last_week(self, date_str: str) -> bool:
        """주어진 날짜가 최근 일주일 이내인지 확인"""
        try:
            date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            one_week_ago = datetime.now() - timedelta(days=7)
            return date >= one_week_ago
        except (ValueError, TypeError):
            return False

    def _format_personal_challenges(self, challenges: List[Dict[str, Any]]) -> str:
        if not challenges:
            return "참여한 개인 챌린지가 없습니다."
        
        formatted = []
        for challenge in challenges:
            status = "성공" if challenge["isSuccess"] else "실패"
            formatted.append(f"- {challenge['title']} ({status})")
        return "\n".join(formatted)

    def _format_group_challenges(self, challenges: List[Dict[str, Any]]) -> str:
        if not challenges:
            return "참여한 단체 챌린지가 없습니다."
        
        formatted = []
        for challenge in challenges:
            # 실천 결과가 있는 챌린지만 필터링
            submissions = challenge.get("submissions", [])
            if not submissions:
                continue

            # 최근 일주일 이내의 제출만 필터링
            recent_submissions = [
                s for s in submissions
                if self._is_within_last_week(s["submittedAt"])
            ]
            
            if not recent_submissions:
                continue

            success_count = sum(1 for s in recent_submissions if s["isSuccess"])
            total_count = len(recent_submissions)
            
            formatted.append(
                f"- {challenge['title']}\n"
                f"  기간: {challenge['startDate']} ~ {challenge['endDate']}\n"
                f"  최근 일주일 성공률: {success_count}/{total_count}"
            )
        return "\n".join(formatted) if formatted else "최근 일주일 동안 참여한 단체 챌린지가 없습니다."

    async def generate_feedback(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # 메모리 정리
            """
            이전 요청의 메모리 잔여물 정리: 다른 요청에서 사용한 메모리가 남아있을 수 있음
            깨끗한 상태에서 시작: 새로운 피드백 생성 전에 메모리를 정리하여 안정성 확보
            메모리 단편화 방지: 연속 요청 시 메모리 단편화 문제 해결
            """
            # shared_model.cleanup_memory()
            
            # 입력 데이터 검증
            if data.get("memberId") is None:  # 0도 유효한 값으로 처리
                return {
                    "status": 400,
                    "message": "memberId는 필수 항목입니다.",
                    "data": None
                }
            
            if not data.get("personalChallenges") and not data.get("groupChallenges"):
                return {
                    "status": 400,
                    "message": "최소 1개의 챌린지 데이터가 필요합니다.",
                    "data": None
                }

            # 챌린지 데이터 포맷팅
            personal_challenges = self._format_personal_challenges(data.get("personalChallenges", []))
            group_challenges = self._format_group_challenges(data.get("groupChallenges", []))

            # 프롬프트 생성
            prompt = self.prompt_template.format(
                personal_challenges=personal_challenges,
                group_challenges=group_challenges
            )

            messages = [
                {"role": "system", "content": "너는 피드백 어시스턴트야."},
                {"role": "user", "content": prompt}
            ]

            try:
                # vLLM(OpenAI 호환) API 호출
                full_feedback = shared_model.get_feedback(
                    messages,
                    temperature=0.7,
                    max_tokens=self.max_tokens
                )

                if not full_feedback.strip():
                    return {
                        "status": 500,
                        "message": "서버 오류로 피드백 생성을 완료하지 못했습니다. 잠시 후 다시 시도해주세요.",
                        "data": None
                    }

                return {
                    "status": 200,
                    "message": "피드백 결과 수신 완료",
                    "data": {
                        "feedback": full_feedback.strip()
                    }
                }

            except Exception as model_error:
                error_trace = traceback.format_exc()
                logger.error(f"Model Error: {str(model_error)}\nTrace: {error_trace}")
                return {
                    "status": 500,
                    "message": "서버 오류로 피드백 결과 저장에 실패했습니다. 잠시 후 다시 시도해주세요.",
                    "data": None
                }

        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(f"General Error: {str(e)}\nTrace: {error_trace}")
            return {
                "status": 500,
                "message": "서버 오류로 피드백 결과 저장에 실패했습니다. 잠시 후 다시 시도해주세요.",
                "data": None
            }
