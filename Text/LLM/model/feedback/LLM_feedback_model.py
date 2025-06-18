from typing import List, Dict, Any, AsyncIterator
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import traceback
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import json
import logging
from huggingface_hub import login
import gc
from Text.LLM.model.chatbot.shared_model import shared_model

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 공유 모델 사용
model = shared_model.model
tokenizer = shared_model.tokenizer

logger.info("Using shared Mistral model for feedback")

class FeedbackModel:
    def __init__(self):
        load_dotenv()
        
        # 공유 모델 사용으로 변경
        self.model = model
        self.tokenizer = tokenizer
        
        logger.info("Feedback model initialized with shared model")
        
        # 한글 기준으로 4-5문장에 적절한 토큰 수로 조정 (약 200-250자)
        self.max_tokens = 200
        # 프롬프트 템플릿을 환경 변수에서 가져오거나 기본값 사용
        self.prompt_template = os.getenv("FEEDBACK_PROMPT_TEMPLATE", """
        다음은 사용자의 챌린지 참여 기록입니다. 이를 바탕으로 긍정적이고 격려하는 피드백을 생성해주세요.

        개인 챌린지:
        {personal_challenges}

        단체 챌린지:
        {group_challenges}

        위 기록을 바탕으로, 사용자의 노력을 인정하고 격려하는 피드백을 생성해주세요.
        실패한 챌린지에 대해서는 위로와 함께 다음 기회를 기대한다는 메시지를 포함해주세요.
        이모지를 적절히 사용하여 친근하고 밝은 톤으로 작성해주세요.
        """)

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
            shared_model.cleanup_memory()
            
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

            try:
                # Mistral 모델을 통한 피드백 생성
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                full_feedback = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                if not full_feedback.strip():
                    return {
                        "status": 500,
                        "message": "서버 오류로 피드백 생성을 완료하지 못했습니다. 잠시 후 다시 시도해주세요.",
                        "data": None
                    }

                callback_url = f"http://34.64.183.21:8080/api/members/feedback/result"
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
        finally:
            # 메모리 정리
            # 성공/실패 관계없이 항상 메모리 정리 실행
            shared_model.cleanup_memory()
            logger.info("Feedback model memory cleanup completed")
