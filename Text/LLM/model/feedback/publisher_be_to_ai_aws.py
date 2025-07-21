import json
import os
import boto3
from dotenv import load_dotenv

load_dotenv()
sqs = boto3.client('sqs', region_name=os.getenv('AWS_DEFAULT_REGION_SERVER2', 'ap-northeast-2'))
queue_url = os.getenv('AWS_SQS_FEEDBACK_QUEUE_URL')

def add_feedback_task(data: dict):
    try:
        message_json = json.dumps(data, ensure_ascii=False)
        response = sqs.send_message(
            QueueUrl=queue_url,
            MessageBody=message_json
        )
        print(f"BE -> AI: feedback_task 발행됨: {data}")
        return response.get('MessageId')
    except Exception as e:
        print(f"피드백 태스크 발행 실패: {e}")
        raise 