
curl -X POST http://localhost:8000/ai/image/verification \
  -H "Content-Type: application/json" \
  -d '{
    "verificationId": 110,
    "type": "GROUP",
    "imageUrl": "https://storage.googleapis.com/leafresh-images/1747987094110-2b84faf8-259d-49c3-8ebb-b96c8cc580ca-camera-capture.jpg",
    "memberId": 112,
    "challengeId": 1,
    "date": "2025-05-24",
    "challengeName": "SNS에 습지 보호 캠페인 알리기",
    "challengeInfo": "2월 2일, 세계 습지의 날을 맞아 습지의 소중함을 더 많은 사람들과 나누는 온라인 캠페인에 함께해요. \n습지 파괴의 문제를 알리고 보호의 필요성을 강조하는 게시물을 SNS에 공유해 주세요. \n해시태그와 함께 자신의 생각이나 사진, 기사 링크 등을 올리며 우리의 관심이 자연을 지키는 힘이 될 수 있음을 알려봐요.\n🔖 추천 해시태그: #세계습지의날 #습지를지키는하루 #습지보호챌린지"
  }'
  