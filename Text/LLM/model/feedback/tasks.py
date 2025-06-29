from Text.LLM.model.feedback.LLM_feedback_model import FeedbackModel
import asyncio
import httpx
import os

def generate_feedback_task(data):
    """
    RQ 워커에서 실행될 피드백 생성 태스크
    비동기 함수를 동기 함수로 래핑하여 RQ에서 실행
    """
    try:
        # FeedbackModel 인스턴스 생성
        feedback_model = FeedbackModel()
        
        # 비동기 함수를 동기적으로 실행
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # 피드백 생성
            feedback_result = loop.run_until_complete(feedback_model.generate_feedback(data))
            
            # 콜백 전송
            if feedback_result and feedback_result.get("status") == 200:
                callback_url = "https://springboot.dev-leafresh.app/api/members/feedback/result"
                callback_payload = {
                    "memberId": data.get("memberId"),
                    "content": feedback_result.get("data", {}).get("feedback", "")
                }
                
                print(f"BE 서비스로 피드백 결과 전송 시도: {callback_url} with payload {callback_payload}")
                
                # 동기적으로 HTTP 요청 전송
                import requests
                response = requests.post(callback_url, json=callback_payload)
                response.raise_for_status()
                print(f"피드백 결과 BE 전송 성공: 상태 코드 {response.status_code}")
                
            elif feedback_result:
                print(f"피드백 모델 오류 발생. 결과를 BE에 전송하지 않습니다. 응답: {feedback_result}")
            else:
                print("피드백 모델 응답이 유효하지 않습니다.")
                
        finally:
            loop.close()
            
    except Exception as e:
        print(f"피드백 생성/전송 중 예상치 못한 오류 발생: {e}")
        raise 