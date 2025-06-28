## 🚀 CoT(Chain of Thought) 테스트 with Langfuse

이 프로젝트는 Langfuse를 사용하여 CoT 방식으로 챗봇 성능을 테스트하고 평가합니다.

### 📋 CoT 테스트란?

CoT(Chain of Thought)는 LLM이 단계별로 사고 과정을 보여주면서 답을 도출하는 방식입니다. 이를 통해:

- **사고 과정의 투명성**: AI가 어떻게 결론에 도달했는지 확인 가능
- **논리적 추론**: 단계별 사고 과정을 통한 정확한 답변
- **성능 평가**: 사고 과정의 품질을 객관적으로 측정

### 🛠️ 설정 방법

1. **Langfuse 계정 생성**
   - [Langfuse Cloud](https://cloud.langfuse.com)에서 계정 생성
   - API 키 발급 (Public Key, Secret Key)

2. **환경 변수 설정**
   ```bash
   # .env 파일에 추가
   LANGFUSE_PUBLIC_KEY=your_langfuse_public_key_here
   LANGFUSE_SECRET_KEY=your_langfuse_secret_key_here
   LANGFUSE_HOST=https://cloud.langfuse.com
   ```

3. **의존성 설치**
   ```bash
   pip install -r requirements.txt
   ```

### 🧪 CoT 테스트 실행

1. **서버 실행**
   ```bash
   uvicorn main:app --port 8000
   ```

2. **CoT 테스트 실행**
   ```bash
   python cot_test_runner.py
   ```

### 📊 평가 기준

CoT 테스트는 다음 4가지 기준으로 평가됩니다:

| 평가 항목 | 가중치 | 설명 |
|-----------|--------|------|
| **사고 과정** | 40% | CoT 키워드 포함 여부, 예상 키워드 매칭 |
| **추론 단계** | 30% | 단계별 사고가 명확한지 |
| **논리적 흐름** | 20% | 논리적 연결어 사용, 자연스러운 흐름 |
| **완성도** | 10% | 응답 길이, 충분한 설명 |

### 📈 결과 확인

1. **콘솔 출력**: 실시간 테스트 결과 확인
2. **Markdown 리포트**: `cot_test_report.md` 파일 생성
3. **Langfuse 대시보드**: 상세한 분석 및 시각화

### 🔍 Langfuse 대시보드 활용

- **Trace 조회**: 각 테스트의 상세 과정 확인
- **성능 분석**: 점수 추이 및 패턴 분석
- **A/B 테스트**: 다양한 프롬프트 비교
- **알림 설정**: 성능 저하 시 자동 알림

### 📁 파일 구조

```
├── langfuse_config.py      # Langfuse 설정 및 평가 로직
├── cot_test_data.py        # CoT 테스트 케이스 및 프롬프트
├── cot_test_runner.py      # 테스트 실행 스크립트
├── cot_test_report.md      # 테스트 결과 리포트 (생성됨)
└── requirements.txt        # 의존성 패키지
```

### 🎯 테스트 케이스

현재 포함된 테스트 케이스:

1. **기본 정보 기반 추천** (`test_001`)
   - 위치: 서울, 직업: 회사원, 카테고리: 에너지절약

2. **자유 텍스트 - 탄소발자국** (`test_002`)
   - "집에서 탄소발자국을 줄이고 싶어요"

3. **자유 텍스트 - 일회용품** (`test_003`)
   - "일회용품 사용을 줄이고 싶은데 어떻게 해야 할까요?"

### 🔧 커스터마이징

새로운 테스트 케이스 추가:

```python
# cot_test_data.py에 추가
{
    "id": "test_004",
    "category": "free_text",
    "input": {
        "message": "새로운 테스트 메시지"
    },
    "expected_keywords": ["키워드1", "키워드2"],
    "cot_prompt": "단계별 사고 과정을 포함한 프롬프트..."
}
```

평가 기준 수정:

```python
# langfuse_config.py의 CoTEvaluator 클래스에서 수정
def evaluate_thought_process(self, trace_id: str, response: str, expected_keywords: List[str]) -> float:
    # 커스텀 평가 로직 구현
    pass
```

### 📞 지원

문제가 있거나 개선 사항이 있으면 이슈를 등록해주세요!

---

## 기존 API 문서

### 엔드포인트

- `POST /ai/chatbot/recommendation/base-info` - 기본 정보 기반 챌린지 추천
- `POST /ai/chatbot/recommendation/free-text` - 자유 텍스트 기반 챌린지 추천
- `POST /ai/censorship` - 콘텐츠 검열
- `POST /ai/feedback` - 피드백 처리
- `GET /health` - 헬스 체크

### 사용 예시

```bash
# 기본 정보 기반 추천
curl -X POST "http://localhost:8000/ai/chatbot/recommendation/base-info" \
  -H "Content-Type: application/json" \
  -d '{
    "location": "서울",
    "workType": "회사원", 
    "category": "에너지절약"
  }'

# 자유 텍스트 기반 추천
curl -X POST "http://localhost:8000/ai/chatbot/recommendation/free-text" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "집에서 탄소발자국을 줄이고 싶어요"
  }'
```