curl -X POST http://localhost:8000/ai/chatbot/recommendation/base-info \
  -H "Content-Type: application/json" \
  -d '{
    "sessionId": "user123",
    "location": "도시",
    "workType": "사무직",
    "category": "제로웨이스트"
  }'


curl -X POST http://localhost:8000/ai/chatbot/recommendation/free-text \
  -H "Content-Type: application/json" \
  -d '{
    "sessionId": "user123",
    "message": "아무거나 추천"
  }'