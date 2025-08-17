import json
import asyncio
import logging
import os
import boto3
from dotenv import load_dotenv
from model.feedback.LLM_feedback_model import FeedbackModel
from model.feedback.publisher_ai_to_be_aws import publish_result

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sqs = boto3.client(
    'sqs',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID_SERVER2'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY_SERVER2'),
    region_name=os.getenv('AWS_DEFAULT_REGION_SERVER2', 'ap-northeast-2')
)
queue_url = os.getenv('AWS_SQS_FEEDBACK_QUEUE_URL')
feedback_model = FeedbackModel()

def run_worker():
    logger.info("🔄 SQS 피드백 워커 시작됨")
    while True:
        try:
            response = sqs.receive_message(
                QueueUrl=queue_url,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=10
            )
            messages = response.get('Messages', [])
            for message in messages:
                try:
                    data = json.loads(message['Body'])
                    logger.info(f"[SQS] 받은 메시지: {data}")
                    feedback_result = asyncio.run(feedback_model.generate_feedback(data))
                    logger.info(f"피드백 생성 결과: {feedback_result}")
                    if feedback_result and feedback_result.get("status") == 200:
                        payload = {
                            "memberId": data.get("memberId"),
                            "content": feedback_result.get("data", {}).get("feedback", ""),
                            "status": "success",
                            "timestamp": data.get("timestamp"),
                            "requestId": data.get("requestId")
                        }
                        message_id = publish_result(payload)
                        if message_id:
                            logger.info(f"[PUBLISH] 피드백 결과 발행 완료 (message ID: {message_id})")
                            logger.info(f"[PUBLISH] Payload: {json.dumps(payload, ensure_ascii=False)}")
                        else:
                            logger.error(f"[PUBLISH] 피드백 결과 발행 실패")
                            logger.error(f"[PUBLISH] Payload: {json.dumps(payload, ensure_ascii=False)}")
                    elif feedback_result and feedback_result.get("status") == 404:
                        logger.warning(f"[404] 피드백 요청을 찾을 수 없음: {feedback_result.get('message')}")
                        error_payload = {
                            "memberId": data.get("memberId"),
                            "error": feedback_result.get("message", "해당 사용자의 피드백 요청을 찾을 수 없습니다."),
                            "status": "not_found",
                            "timestamp": data.get("timestamp"),
                            "requestId": data.get("requestId")
                        }
                        message_id = publish_result(error_payload)
                        if message_id:
                            logger.warning(f"[PUBLISH] 404 오류 발행 (message ID: {message_id})")
                            logger.warning(f"[PUBLISH] 404 Payload: {json.dumps(error_payload, ensure_ascii=False)}")
                        else:
                            logger.error(f"[PUBLISH] 404 오류 발행 실패")
                            logger.error(f"[PUBLISH] 404 Payload: {json.dumps(error_payload, ensure_ascii=False)}")
                    else:
                        error_payload = {
                            "memberId": data.get("memberId"),
                            "error": feedback_result.get("message", "Unknown error") if feedback_result else "No response from model",
                            "status": "error",
                            "timestamp": data.get("timestamp"),
                            "requestId": data.get("requestId")
                        }
                        message_id = publish_result(error_payload)
                        if message_id:
                            logger.error(f"[PUBLISH] 피드백 오류 발행 (message ID: {message_id})")
                            logger.error(f"[PUBLISH] Error Payload: {json.dumps(error_payload, ensure_ascii=False)}")
                        else:
                            logger.error(f"[PUBLISH] 피드백 오류 발행 실패")
                            logger.error(f"[PUBLISH] Error Payload: {json.dumps(error_payload, ensure_ascii=False)}")
                    # 메시지 삭제(ACK)
                    sqs.delete_message(
                        QueueUrl=queue_url,
                        ReceiptHandle=message['ReceiptHandle']
                    )
                    logger.info("[ACK] 메시지 처리 완료 및 삭제")
                except json.JSONDecodeError as e:
                    logger.error(f"[ERROR] JSON 파싱 실패: {e}")
                    # JSON 파싱 실패한 메시지는 삭제 (재시도해도 소용없음)
                    sqs.delete_message(
                        QueueUrl=queue_url,
                        ReceiptHandle=message['ReceiptHandle']
                    )
                except Exception as e:
                    logger.error(f"[ERROR] 피드백 처리 실패: {e}")
                    # 메시지 삭제하지 않음(재시도)
        except Exception as e:
            logger.error(f"[ERROR] SQS 연결 또는 메시지 수신 실패: {e}")
            # 연결 실패 시 잠시 대기 후 재시도
            import time
            time.sleep(5)
            continue
        
        # 정상적으로 처리된 경우 짧은 대기
        import time
        time.sleep(1)

if __name__ == "__main__":
    run_worker() 