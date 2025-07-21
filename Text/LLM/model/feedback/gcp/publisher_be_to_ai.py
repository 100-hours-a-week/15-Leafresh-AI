from google.cloud import pubsub_v1
import json
import os
from dotenv import load_dotenv

load_dotenv()

project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "leafresh-dev2")
topic_id = os.getenv("PUBSUB_TOPIC_FEEDBACK_DEV")  # leafresh-feedback-topic

publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(project_id, topic_id)

def add_feedback_task(data: dict):
    """
    피드백 생성 요청을 토픽에 발행
    Args:
        data: 피드백 생성에 필요한 데이터
    """
    try:
        message_json = json.dumps(data, ensure_ascii=False)
        future = publisher.publish(topic_path, message_json.encode("utf-8"))
        print(f"BE -> AI: feedback_task 발행됨: {data}")
        return future.result()
    except Exception as e:
        print(f"피드백 태스크 발행 실패: {e}")
        raise 