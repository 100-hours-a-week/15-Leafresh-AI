import json
import os
import boto3
from dotenv import load_dotenv

load_dotenv()
sqs = boto3.client('sqs', region_name=os.getenv('AWS_DEFAULT_REGION_SERVER2', 'ap-northeast-2'))
queue_url = os.getenv('AWS_SQS_FEEDBACK_RESULT_QUEUE_URL')

def publish_result(data: dict):
    message_json = json.dumps(data)
    try:
        response = sqs.send_message(
            QueueUrl=queue_url,
            MessageBody=message_json
        )
        return response.get('MessageId')
    except Exception as e:
        print(f"[ERROR] 피드백 처리 실패: {e}") 