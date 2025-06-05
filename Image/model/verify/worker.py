from google.cloud import pubsub_v1
from model.verify.LLM_verify_model import ImageVerifyModel
import json
import requests

from dotenv import load_dotenv
import os

load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
subscription_id = os.getenv("PUBSUB_SUBSCRIPTION")

subscription_path = f"projects/{project_id}/subscriptions/{subscription_id}"

subscriber = pubsub_v1.SubscriberClient()
verifier = ImageVerifyModel()

def run_worker():
    def callback(message):
        print("callback 메시지 수신됨", message)
        try:
            data = json.loads(message.data.decode("utf-8"))

            blob_name = data["imageUrl"].split("/")[-1]
            challenge_type = data["type"]
            challenge_id = int(data["challengeId"])
            challenge_name = data["challengeName"]

            result = verifier.image_verify(os.getenv("BUCKET_NAME"), blob_name, challenge_type, challenge_id, challenge_name)
            
            # 로깅용
            print(f"인증 결과: {result}")

            # '예' 여부가 정확히 일치할 때만 True
            # is_verified = result.strip() == "예"
            is_verified = "예" in result.strip().lower()

            # 콜백 URL 내 challengeId 치환
            # -> CALLBACK_URL에 {verificationId}가 포함되는 경우, Python에서 실제 전송 전에 .format() 또는 f-string으로 치환해줘야함 
            formatted_url = os.getenv("CALLBACK_URL_VERIFY").format(verificationId=data["verificationId"])

            
            # 결과 콜백 전송
            requests.post(formatted_url, json={
                "type": data["type"],
                "memberId": data["memberId"],
                "challengeId": data["challengeId"],
                "date": data["date"],
                "result": is_verified
            })
            
            message.ack()

        except Exception as e:
            print(f"[ERROR] 처리 실패: {e}")
            message.nack()

    # 구독 시작
    future = subscriber.subscribe(subscription_path, callback=callback)
    print(f"Listening on {subscription_path}...")
    future.result()