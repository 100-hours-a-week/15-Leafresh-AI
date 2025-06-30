from google.cloud import pubsub_v1
import json, os

from dotenv import load_dotenv

load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
topic_id = os.getenv("PUBSUB_TOPIC_AI_TO_BE")

publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(project_id, topic_id)

def publish_result(data: dict):
    message_json = json.dumps(data)
    future = publisher.publish(topic_path, message_json.encode("utf-8"))

    print("AI -> BE : publich_message 발행됨: ", data)
    return future.result()