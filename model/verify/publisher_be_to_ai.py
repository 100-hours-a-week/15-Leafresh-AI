import boto3, json, os
from dotenv import load_dotenv

load_dotenv()

# SQS 클라이언트 초기화
sqs = boto3.client("sqs",
                   aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                   aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                   region_name=os.getenv("AWS_DEFAULT_REGION"))

verify_queue_url = os.getenv("AWS_SQS_INPUT_QUEUE_URL")

def add_task(data: dict):
    try:
        message_json = json.dumps(data)
        response = sqs.send_message(QueueUrl = verify_queue_url, MessageBody = message_json, MessageGroupId="image-verification")

        message_id = response["MessageId"]
        print("[PUB] BE -> AI : SQS 메시지 발행됨 :", data)
        print("[PUB] BE -> AI : SQS 메시지 ID :", message_id)
        return message_id

    except Exception as e:
        print("[ERROR] BE -> AI : SQS 메시지 발행 실패 :", e)
        return None