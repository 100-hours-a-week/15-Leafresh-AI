import boto3, json, os, time
from dotenv import load_dotenv

from model.verify.LLM_verify_model import ImageVerifyModel
from model.verify.publisher_ai_to_be_aws import publish_result


load_dotenv()

# SQS 클라이언트 초기화
sqs = boto3.client("sqs",
                   aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID_SERVER1"),
                   aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY_SERVER1"),
                   region_name=os.getenv("AWS_DEFAULT_REGION_SERVER1"))

verify_queue_url = os.getenv("AWS_SQS_INPUT_QUEUE_URL")
bucket_name = os.getenv("BUCKET_NAME_DEV")

verifier = ImageVerifyModel()


# SQS 메시지 처리 및 결과 발행 
def process_message(data: dict):
    try:
        blob_name = data["imageUrl"].split("/")[-1]
        challenge_type = data["type"]
        challenge_id = int(data["challengeId"])
        challenge_name = data["challengeName"]
        challenge_info = data["challengeInfo"]

        result = verifier.image_verify(bucket_name, blob_name, challenge_type, challenge_id, challenge_name, challenge_info)

        print(f"인증 결과: {result}")

        # '예' 여부가 정확히 일치할 때만 True
        is_verified = "예" in result.strip().lower()

        # 결과 콜백 전송
        payload = {
            "type": data["type"],
            "memberId": data["memberId"],
            "challengeId": data["challengeId"],
            "verificationId": data["verificationId"],
            "date": data["date"],
            "result": is_verified
        }

        message_id = publish_result(payload)

        print(f"[PUB] 인증 결과 발행 완료 (message ID: {message_id})")
        print(f"[PUB] Payload: {json.dumps(payload, ensure_ascii=False)}")
    except Exception as e:
        print(f"[ERROR] 메시지 처리 실패: {e}")
        raise


def run_worker():
    print(f"[INFO] BE -> AI : SQS 구독 시작: Listening to {verify_queue_url}...")

    while True:
        try:
            response = sqs.receive_message(
                QueueUrl = verify_queue_url,
                MaxNumberOfMessages = 1,        # 한 번에 1개의 메시지만 처리
                WaitTimeSeconds = 10,           # 최대 10초 대기 
            )

            for message in response.get("Messages", []):
                print("[SUB] BE -> AI : SQS 메시지 수신됨")

                try:
                    data = json.loads(message["Body"])
                    print(f"[SUB] BE -> AI : SQS 메시지 내용 : {data}")

                    # 메시지 처리
                    process_message(data)

                    # 메시지 삭제
                    sqs.delete_message(
                        QueueUrl=verify_queue_url,
                        ReceiptHandle=message["ReceiptHandle"]
                    )
                    print("[INFO] BE -> AI : SQS 메시지 삭제 완료 (ACK)")

                except Exception:
                    print("[ERROR] 메시지 처리 중 오류 발생, 메시지 삭제하지 않음")     # 삭제하지 않으면 SQS가 자동으로 재시도 

        except Exception as e:
            print(f"[ERROR] SQS 메시지 수신 중 오류 발생 | SQS Polling 실패 : {e}")
            time.sleep(1)
                    
