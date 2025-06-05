
curl -X POST http://localhost:8000/ai/challenges/group/validation \
  -H "Content-Type: application/json" \
  -d '{
    "memberId": 12345,
    "challengeName": "텀블러 챌린지",
    "startDate": "2025-04-20",
    "endDate": "2025-04-27",
    "challenge": [
      {
        "id": 200,
        "name": "플로깅",
        "startDate": "2025-01-01",
        "endDate": "2025-06-30"
      },
      {
        "id": 201,
        "name": "비건 식단",
        "startDate": "2025-01-01",
        "endDate": "2025-06-30"
      }
    ]
  }'


curl -X POST http://localhost:8000/ai/challenges/group/validation \
  -H "Content-Type: application/json" \
  -d '{
    "memberId": 54321,
    "challengeName": "노아의 챌린지",
    "startDate": "2025-04-20",
    "endDate": "2025-04-27",
    "challenge": []
  }'


curl -X POST http://localhost:8000/ai/challenges/group/validation \
  -H "Content-Type: application/json" \
  -d '{
    "memberId": 11,
    "challengeName": "텀블러 이용 챌린지",
    "startDate": "2025-04-20",
    "endDate": "2025-04-27",
    "challenge": [
      { "id": 200, "name": "텀블러 챌린지", "startDate": "2025-04-20", "endDate": "2025-04-27" },
      { "id": 201, "name": "플로깅 챌린지", "startDate": "2025-04-20", "endDate": "2025-04-27" },
      { "id": 202, "name": "플로깅 챌린지", "startDate": "2025-05-20", "endDate": "2025-05-27" },
      { "id": 203, "name": "비건 식단 챌린지", "startDate": "2025-05-20", "endDate": "2025-05-27" },
      { "id": 204, "name": "이메일 휴지통 10개 비우기 챌린지", "startDate": "2025-05-14", "endDate": "2025-05-20" }
    ]
  }'


curl -X POST http://localhost:8000/ai/challenges/group/validation \
  -H "Content-Type: application/json" \
  -d '{
    "memberId": 46,
    "challengeName": "플라스틱 대신 유리병 사용하기",
    "startDate": "2025-08-01",
    "endDate": "2025-08-07",
    "challenge": [
      { "id": 200, "name": "텀블러 챌린지", "startDate": "2025-04-20", "endDate": "2025-04-27" },
      { "id": 201, "name": "플로깅 챌린지", "startDate": "2025-04-20", "endDate": "2025-04-27" },
      { "id": 202, "name": "플로깅 챌린지", "startDate": "2025-05-20", "endDate": "2025-05-27" },
      { "id": 203, "name": "비건 식단 챌린지", "startDate": "2025-05-20", "endDate": "2025-05-27" },
      { "id": 204, "name": "이메일 휴지통 10개 비우기 챌린지", "startDate": "2025-05-14", "endDate": "2025-05-20" },
      { "id": 205, "name": "점심 시간 도시락 싸먹기 챌린지", "startDate": "2025-05-01", "endDate": "2025-05-31" },
      { "id": 206, "name": "텀블러 매일 사용하기", "startDate": "2025-08-01", "endDate": "2025-08-07" },
      { "id": 207, "name": "대중교통 이용하기", "startDate": "2025-04-20", "endDate": "2025-04-27" },
      { "id": 208, "name": "종이 타월 사용 줄이기", "startDate": "2025-05-10", "endDate": "2025-05-20" },
      { "id": 209, "name": "하루에 한 번 이상 환경을 생각하는 행동 하기", "startDate": "2025-06-01", "endDate": "2025-06-07" },
      { "id": 210, "name": "종이컵 대신 머그컵", "startDate": "2025-06-10", "endDate": "2025-06-17" },
      { "id": 211, "name": "매일 식물에게 물 주기", "startDate": "2025-07-01", "endDate": "2025-07-07" },
      { "id": 212, "name": "대중교통 이용하기", "startDate": "2025-07-20", "endDate": "2025-07-27" },
      { "id": 213, "name": "🌱 환경을 위한 작은 실천 🌎", "startDate": "2025-06-01", "endDate": "2025-06-07" },
      { "id": 214, "name": "Bring Your Own Cup", "startDate": "2025-06-15", "endDate": "2025-06-21" },
      { "id": 215, "name": "하루에 한 번 채식하기", "startDate": "2025-06-10", "endDate": "2025-06-17" },
      { "id": 216, "name": "#환경보호#실천#텀블러", "startDate": "2025-06-15", "endDate": "2025-06-22" },
      { "id": 217, "name": "플라스틱 대신 유리병 사용하기", "startDate": "2025-06-01", "endDate": "2025-06-07" },
      { "id": 218, "name": "하루 동안 종이컵 대신 텀블러만 사용해보기", "startDate": "2025-07-10", "endDate": "2025-07-17" }
    ]
  }'


curl -X POST http://localhost:8000/ai/challenges/group/validation \
  -H "Content-Type: application/json" \
  -d '{
    "memberId": 40,
    "challengeName": "하루에 한 번 채식하기",
    "startDate": "2025-06-10",
    "endDate": "2025-06-17",
    "challenge": [
      { "id": 200, "name": "텀블러 챌린지", "startDate": "2025-04-20", "endDate": "2025-04-27" },
      { "id": 201, "name": "플로깅 챌린지", "startDate": "2025-04-20", "endDate": "2025-04-27" },
      { "id": 202, "name": "플로깅 챌린지", "startDate": "2025-05-20", "endDate": "2025-05-27" },
      { "id": 203, "name": "비건 식단 챌린지", "startDate": "2025-05-20", "endDate": "2025-05-27" },
      { "id": 204, "name": "이메일 휴지통 10개 비우기 챌린지", "startDate": "2025-05-14", "endDate": "2025-05-20" },
      { "id": 205, "name": "점심 시간 도시락 싸먹기 챌린지", "startDate": "2025-05-01", "endDate": "2025-05-31" },
      { "id": 206, "name": "텀블러 매일 사용하기", "startDate": "2025-08-01", "endDate": "2025-08-07" },
      { "id": 207, "name": "대중교통 이용하기", "startDate": "2025-04-20", "endDate": "2025-04-27" },
      { "id": 208, "name": "종이 타월 사용 줄이기", "startDate": "2025-05-10", "endDate": "2025-05-20" },
      { "id": 209, "name": "하루에 한 번 이상 환경을 생각하는 행동 하기", "startDate": "2025-06-01", "endDate": "2025-06-07" },
      { "id": 210, "name": "종이컵 대신 머그컵", "startDate": "2025-06-10", "endDate": "2025-06-17" },
      { "id": 211, "name": "매일 식물에게 물 주기", "startDate": "2025-07-01", "endDate": "2025-07-07" },
      { "id": 212, "name": "대중교통 이용하기", "startDate": "2025-07-20", "endDate": "2025-07-27" },
      { "id": 213, "name": "🌱 환경을 위한 작은 실천 🌎", "startDate": "2025-06-01", "endDate": "2025-06-07" },
      { "id": 214, "name": "Bring Your Own Cup", "startDate": "2025-06-15", "endDate": "2025-06-21" }
    ]
  }'

