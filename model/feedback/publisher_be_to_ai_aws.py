import json
import os
import boto3
from dotenv import load_dotenv
import uuid

load_dotenv()

# SQS 클라이언트 설정 (인증 정보 포함)
sqs = boto3.client(
    'sqs',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID_SERVER2'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY_SERVER2'),
    region_name=os.getenv('AWS_DEFAULT_REGION_SERVER2', 'ap-northeast-2')
)
queue_url = os.getenv('AWS_SQS_FEEDBACK_QUEUE_URL')

def add_feedback_task(data: dict):
    try:
        message_json = json.dumps(data, ensure_ascii=False)
        
        # FIFO Queue인지 확인 (URL에 .fifo가 포함되어 있는지)
        is_fifo_queue = '.fifo' in queue_url if queue_url else False
        
        if is_fifo_queue:
            # FIFO Queue인 경우 MessageGroupId와 MessageDeduplicationId 필요
            message_group_id = str(data.get("memberId", "default"))
            message_deduplication_id = str(uuid.uuid4())
            
            response = sqs.send_message(
                QueueUrl=queue_url,
                MessageBody=message_json,
                MessageGroupId=message_group_id,
                MessageDeduplicationId=message_deduplication_id
            )
        else:
            # Standard Queue인 경우 기본 설정
            response = sqs.send_message(
                QueueUrl=queue_url,
                MessageBody=message_json
            )
        
        message_id = response.get('MessageId')
        print(f"[SUCCESS] BE -> AI: feedback_task 발행 완료 (MessageId: {message_id})")
        print(f"[SUCCESS] Data: {data}")
        return message_id
    except Exception as e:
        print(f"[ERROR] 피드백 태스크 발행 실패: {e}")
        print(f"[ERROR] Queue URL: {queue_url}")
        print(f"[ERROR] Data: {data}")
        raise 