# Feedback Router Changes 

# 2024-06-15

## 주요 변경사항

### 1. 날짜 처리 개선
- `Submission` 모델에 `submittedAt` 필드의 커스텀 밸리데이터 추가
- 배열 형식의 날짜 데이터 처리 지원 ([year, month, day, hour, minute, second, microsecond])
- 마이크로초가 999999를 초과하는 경우 자동 조정

### 2. `GroupChallenge` 모델 개선
- `startDate`와 `endDate` 필드에 커스텀 밸리데이터 추가
- 배열 형식의 날짜 데이터 처리 지원 ([year, month, day, hour, minute, second])

### 3. 백엔드 통신 개선
- 콜백 URL 수정: `https://springboot.dev-leafresh.app/api/members/feedback/result`
- 콜백 요청 시 인증 헤더 추가
- URL 중복 슬래시(//) 제거

### 4. 에러 처리 강화
- 상세한 에러 로깅 추가
- HTTP 상태 코드별 적절한 에러 메시지 반환
- 백엔드 통신 실패 시 상세한 에러 정보 로깅

### 5. 응답 형식 표준화
- 모든 응답에 status, message, data 필드 포함
- 202 Accepted 응답 추가 (비동기 처리 시작 시)
- 400 Bad Request 응답 개선 (유효성 검사 실패 시)

## 기술적 세부사항

### 날짜 처리 로직
```python
@field_validator('submittedAt', mode='before')
@classmethod
def parse_submitted_at(cls, v):
    if isinstance(v, list):
        if len(v) == 7:
            microsecond = min(v[6], 999999)
            return datetime(v[0], v[1], v[2], v[3], v[4], v[5], microsecond)
        return datetime(*v)
    return v
```

### 백엔드 통신 로직
```python
callback_url = f"https://springboot.dev-leafresh.app/api/members/feedback/result"
callback_payload = {
    "memberId": data.get("memberId"),
    "content": feedback_result.get("data", {}).get("feedback", "")
}
```

## 테스트 방법
1. 날짜 형식 테스트:
   - 배열 형식: `[2024, 6, 15, 12, 0, 0, 0]`
   - ISO 문자열 형식: `"2024-06-15T12:00:00Z"`

2. 백엔드 통신 테스트:
   - 피드백 생성 요청
   - 콜백 응답 확인
   - 에러 케이스 처리 확인