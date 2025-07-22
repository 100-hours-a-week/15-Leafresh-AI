from google.cloud import pubsub_v1
import json
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

project_id = os.getenv("GOOGLE_CLOUD_PROJECT_MAC")
topic_id = os.getenv("PUBSUB_TOPIC_FEEDBACK_RESULT_DEV")

publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(project_id, topic_id)

def publish_result(data: dict):
    """
    피드백 결과를 토픽에 발행
    Args:
        data: 발행할 데이터
    """
    message_json = json.dumps(data)
    try:
        future = publisher.publish(topic_path, message_json.encode("utf-8"))
        return future.result()
    except Exception as e:
        print(f"[ERROR] 피드백 처리 실패: {e}") 
        # DLQ 발행 코드 완전히 제거!
        # 그냥 에러만 로그 
