curl -X POST http://localhost:8000/api/members/feedback/result \
  -H "Content-Type: application/json" \
  -d '{
    "memberId": 4,
    "personalChallenges": [],
    "groupChallenges": [
      {
        "id": 24,
        "title": "텀블러 챌린지",
        "startDate": "2025-06-12T00:00:00",
        "endDate": "2025-06-13T23:59:59",
        "submissions": []
      },
      {
        "id": 23,
        "title": "123123",
        "startDate": "2025-06-12T00:00:00",
        "endDate": "2025-06-14T23:59:59",
        "submissions": [
          {
            "isSuccess": true,
            "submittedAt": "2025-06-07T13:08:16.710986"
          }
        ]
      },
      {
        "id": 22,
        "title": "string213123123",
        "startDate": "2025-06-05T00:00:00",
        "endDate": "2025-06-20T23:59:59",
        "submissions": []
      },
      {
        "id": 21,
        "title": "string",
        "startDate": "2025-06-05T00:00:00",
        "endDate": "2025-06-20T23:59:59",
        "submissions": []
      },
      {
        "id": 18,
        "title": "제로웨이스트",
        "startDate": "2025-06-12T00:00:00",
        "endDate": "2025-06-19T23:59:59",
        "submissions": []
      },
      {
        "id": 20,
        "title": "string",
        "startDate": "2025-06-05T00:00:00",
        "endDate": "2025-06-20T23:59:59",
        "submissions": []
      }
    ]
  }'