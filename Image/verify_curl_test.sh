
curl -X POST http://localhost:8000/ai/image/verification/stream \
  -H "Content-Type: application/json" \
  -d '{
    "verificationId": 110,
    "type": "GROUP",
    "imageUrl": "https://storage.googleapis.com/leafresh-images/1747987094110-2b84faf8-259d-49c3-8ebb-b96c8cc580ca-camera-capture.jpg",
    "memberId": 112,
    "challengeId": 1,
    "date": "2025-05-24",
    "challengeName": "SNS에 습지 보호 캠페인 알리기"
  }'