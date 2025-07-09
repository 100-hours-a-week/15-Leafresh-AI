from google.cloud import pubsub_v1
from model.feedback.LLM_feedback_model import FeedbackModel
from model.feedback.publisher_ai_to_be import publish_result
import json
import asyncio
import logging
from dotenv import load_dotenv
import os

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_MAC")

# 피드백 요청 구독용 (dev2 프로젝트) - BE가 발행하는 곳
feedback_sub_project = "leafresh-dev2"  # BE가 발행하는 프로젝트
feedback_subscription_id = os.getenv("PUBSUB_SUBSCRIPTION_FEEDBACK_DEV")

# 피드백 결과 발행용 (mac 프로젝트)
feedback_pub_project = os.getenv("GOOGLE_CLOUD_PROJECT_MAC")
feedback_result_topic_id = os.getenv("PUBSUB_TOPIC_FEEDBACK_RESULT_DEV")

subscription_path = f"projects/{feedback_sub_project}/subscriptions/{feedback_subscription_id}"

subscriber = pubsub_v1.SubscriberClient()
feedback_model = FeedbackModel()

def run_worker():
    def callback(message):
        logger.info("[CALLBACK] 피드백 요청 메시지 수신됨")
        
        try:
            # 메시지 파싱
            data = json.loads(message.data.decode("utf-8"))
            logger.info(f"[DEBUG] 파싱된 data dict: {data}")
            
            # 피드백 생성
            feedback_result = asyncio.run(feedback_model.generate_feedback(data))
            logger.info(f"피드백 생성 결과: {feedback_result}")
            
            if feedback_result and feedback_result.get("status") == 200:
                # 성공 결과 발행
                payload = {
                    "memberId": data.get("memberId"),
                    "content": feedback_result.get("data", {}).get("feedback", ""),
                    "status": "success",
                    "timestamp": data.get("timestamp"),
                    "requestId": data.get("requestId")
                }
                
                message_id = publish_result(payload)
                logger.info(f"[PUBLISH] 피드백 결과 발행 완료 (message ID: {message_id})")
                logger.info(f"[PUBLISH] Payload: {json.dumps(payload, ensure_ascii=False)}")
                
            elif feedback_result and feedback_result.get("status") == 404:
                # 404 오류: 피드백 요청을 찾을 수 없는 경우
                logger.warning(f"[404] 피드백 요청을 찾을 수 없음: {feedback_result.get('message')}")
                error_payload = {
                    "memberId": data.get("memberId"),
                    "error": feedback_result.get("message", "해당 사용자의 피드백 요청을 찾을 수 없습니다."),
                    "status": "not_found",
                    "timestamp": data.get("timestamp"),
                    "requestId": data.get("requestId")
                }
                
                message_id = publish_result(error_payload)
                logger.warning(f"[PUBLISH] 404 오류 발행 (message ID: {message_id})")
                logger.warning(f"[PUBLISH] 404 Payload: {json.dumps(error_payload, ensure_ascii=False)}")
                
            else:
                # 기타 실패 결과 발행
                error_payload = {
                    "memberId": data.get("memberId"),
                    "error": feedback_result.get("message", "Unknown error") if feedback_result else "No response from model",
                    "status": "error",
                    "timestamp": data.get("timestamp"),
                    "requestId": data.get("requestId")
                }
                
                message_id = publish_result(error_payload)
                logger.error(f"[PUBLISH] 피드백 오류 발행 (message ID: {message_id})")
                logger.error(f"[PUBLISH] Error Payload: {json.dumps(error_payload, ensure_ascii=False)}")
            
            # 메시지 확인 (ACK)
            message.ack()
            logger.info("[ACK] 메시지 처리 완료 및 확인")
            
        except json.JSONDecodeError as e:
            logger.error(f"[ERROR] JSON 파싱 실패: {e}")
            # JSON 파싱 실패 시 DLQ로 발행
            error_payload = {
                "error": f"Invalid JSON format: {str(e)}",
                "status": "error",
                "rawData": message.data.decode("utf-8")[:500]  # 원본 데이터 일부 포함
            }
            try:
                message_id = publish_result(error_payload)
                logger.error(f"[PUBLISH] JSON 오류 발행 (message ID: {message_id})")
            except Exception as publish_error:
                logger.error(f"[ERROR] JSON 오류 발행 실패: {publish_error}")
            
            message.nack()  # 재시도 요청
            
        except Exception as e:
            logger.error(f"[ERROR] 피드백 처리 실패: {e}")
            # 일반 오류 시 발행
            error_payload = {
                "error": f"Processing error: {str(e)}",
                "status": "error",
                "memberId": data.get("memberId") if 'data' in locals() else None
            }
            try:
                message_id = publish_result(error_payload)
                logger.error(f"[PUBLISH] 처리 오류 발행 (message ID: {message_id})")
            except Exception as publish_error:
                logger.error(f"[ERROR] 처리 오류 발행 실패: {publish_error}")
            
            message.nack()  # 재시도 요청

    # 구독 시작
    logger.info(f"[SUB] 구독 시작 시도: Listening on {subscription_path}...")
    future = subscriber.subscribe(subscription_path, callback=callback)
    logger.info("[DEBUG] subscriber.subscribe() 호출 완료")
    
    try:
        future.result()  # 무한 대기
    except KeyboardInterrupt:
        logger.info("[INFO] 워커 종료 요청됨")
        future.cancel()
    except Exception as e:
        logger.error(f"[ERROR] 구독 중 오류 발생: {e}")
        future.cancel()

if __name__ == "__main__":
    run_worker() 