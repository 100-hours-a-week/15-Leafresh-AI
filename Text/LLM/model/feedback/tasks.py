import asyncio
import httpx
import os
import requests
from Text.LLM.model.feedback.LLM_feedback_model import FeedbackModel
from Text.LLM.model.chatbot.shared_model import shared_model

feedback_model = FeedbackModel()

def generate_feedback_task(data):
    """
    RQ 워커에서 실행될 피드백 생성 태스크
    FastAPI 서버에서 이미 로드된 shared_model을 재사용
    """
    try:
        # FastAPI 서버에서 이미 로드된 shared_model 사용
        # 새로운 FeedbackModel 인스턴스 생성 (shared_model을 내부적으로 사용)
        feedback_model = FeedbackModel()
        
        # 피드백 생성
        feedback_result = asyncio.run(feedback_model.generate_feedback(data))
        
        # 콜백 전송
        if feedback_result and feedback_result.get("status") == 200:
            callback_url = "http://34.64.183.21:8080/api/members/feedback/result"
            callback_payload = {
                "memberId": data.get("memberId"),
                "content": feedback_result.get("data", {}).get("feedback", "")
            }
            
            print(f"BE 서비스로 피드백 결과 전송 시도: {callback_url} with payload {callback_payload}")
            
            # 동기적으로 HTTP 요청 전송
            callback_response = requests.post(callback_url, json=callback_payload)
            callback_response.raise_for_status()
            print(f"피드백 결과 BE 전송 성공: 상태 코드 {callback_response.status_code}")
            
        elif feedback_result:
            print(f"피드백 모델 오류 발생. 결과를 BE에 전송하지 않습니다. 응답: {feedback_result}")
        else:
            print("피드백 모델 응답이 유효하지 않습니다.")
            
    except Exception as e:
        print(f"피드백 생성/전송 중 예상치 못한 오류 발생: {e}")
        raise 