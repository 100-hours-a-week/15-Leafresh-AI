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

# 2025-06-26

## 주요 변경사항: Redis 큐 기반 비동기 피드백 생성 전환

### 1. Redis 큐(RQ) 기반 비동기 처리 도입
- FastAPI 서버에서 피드백 생성 요청을 받으면, 동기/백그라운드 처리 대신 **Redis 큐(feedback)**에 작업을 등록하도록 변경
- 별도의 **RQ 워커**가 Redis 큐에서 작업을 꺼내 LLM 피드백 생성 및 콜백 전송을 담당
- 서버 확장성, 장애 복구, 대량 트래픽 대응력 향상

### 2. 코드 구조 요약
- `feedback_router.py`에서 요청을 받으면 `feedback_queue.enqueue(generate_feedback_task, data)`로 큐에 등록
- `tasks.py`의 `generate_feedback_task` 함수가 실제 LLM 피드백 생성 및 콜백 전송을 담당
- RQ 워커는 별도 프로세스(`rq worker feedback`)로 실행

```mermaid
graph TD
    A[클라이언트 요청] --> B[FastAPI 서버]
    B --> C[Redis 큐에 작업 등록]
    B --> D[즉시 202 응답 반환]
    C --> E[RQ 워커가 큐에서 작업 꺼냄]
    E --> F[LLM 피드백 생성]
    F --> G[BE 서버로 콜백 전송]
```


### 3. Redis 서버 관리법
#### Redis 서버 켜는 법
- Ubuntu 기준:
  ```bash
  sudo service redis-server start
  # 또는
  redis-server
  ```
- Docker 사용 시:
  ```bash
  docker run -d --name redis-queue -p 6379:6379 redis:latest
  ```

#### Redis 서버 끄는 법
- Ubuntu 기준:
  ```bash
  sudo service redis-server stop
  ```
- Docker 사용 시:
  ```bash
  docker stop redis-queue
  docker rm redis-queue
  ```

#### Redis 서버 상태 확인
```bash
redis-cli ping
# → PONG 이 나오면 정상 동작
```

### 4. RQ 워커 실행/중지
- 워커 실행:
  ```bash
  rq worker feedback
  ```
- 워커는 여러 개 띄울 수 있음(동시 처리량 증가)
- 워커 중지는 Ctrl+C 또는 프로세스 종료

### 5. 전체 구조 요약
- FastAPI 서버: 요청을 Redis 큐에 등록 → 즉시 202 응답 반환
- RQ 워커: 큐에서 작업을 꺼내 LLM 피드백 생성 및 콜백 전송
- Redis 서버: 큐 역할, AI 서버와 워커가 함께 사용

### 6. 테스트 및 운영 팁
- Redis, 워커, FastAPI 서버가 모두 실행 중이어야 정상 동작
- 장애 복구, 확장성, 분산 처리에 유리
- 운영 환경에서는 Redis 보안 설정, 모니터링, 백업 등 추가 고려 필요

# 2025-06-29

## 주요 변경사항: 통합 모델 아키텍처 최적화

### 1. 모델 서비스 분리 제거 및 통합 구조 채택
- 기존에 제안했던 별도 모델 서비스(`model_service.py`) 제거
- **shared_model** 싱글톤 패턴을 통한 통합 모델 관리 구조 채택
- FastAPI 서버 시작 시 4비트 양자화된 Mistral 모델이 메모리에 로드되어 모든 기능에서 공유

### 2. 현재 아키텍처 구조
```
FastAPI 서버 (포트 8000)
    ↓
shared_model (4비트 양자화된 Mistral-7B-Instruct-v0.3)
    ↓
├── 챗봇 기능 (base-info, free-text)
├── 피드백 기능 (feedback generation)
└── 검열 기능 (별도 모델 사용)
```

### 3. FastAPI 서버 실행 방법

#### 3.1 기본 실행
```bash
# 프로젝트 루트 디렉토리에서
cd /home/ubuntu/15-Leafresh-AI/Text/LLM

# FastAPI 서버 실행
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### 3.2 백그라운드 실행
```bash
# 백그라운드에서 실행
nohup uvicorn main:app --host 0.0.0.0 --port 8000 > fastapi.log 2>&1 &

# 프로세스 확인
ps aux | grep uvicorn

# 로그 확인
tail -f fastapi.log
```

#### 3.3 서버 중지
```bash
# 프로세스 ID 찾기
ps aux | grep uvicorn

# 프로세스 종료
kill <process_id>

# 또는 강제 종료
kill -9 <process_id>
```

### 4. 전체 시스템 실행 순서

#### 4.1 필수 서비스 시작 순서
```bash
# 1. Redis 서버 시작
sudo service redis-server start

# 2. FastAPI 서버 시작 (모델 자동 로드)
cd /home/ubuntu/15-Leafresh-AI/Text/LLM
uvicorn main:app --host 0.0.0.0 --port 8000

# 3. RQ 워커 시작 (새 터미널에서)
cd /home/ubuntu/15-Leafresh-AI/Text/LLM
rq worker feedback
```

#### 4.2 서비스 상태 확인
```bash
# Redis 상태 확인
redis-cli ping

# FastAPI 서버 상태 확인
curl http://localhost:8000/docs

# RQ 워커 상태 확인 (로그에서 확인)
```

### 5. 모델 로딩 최적화

#### 5.1 지연 로딩 (Lazy Loading)
- FastAPI 서버 시작 시 모델이 즉시 로드되지 않음
- 첫 번째 요청 시에만 모델 로드 (메모리 효율성)
- `shared_model`의 `@property` 데코레이터를 통한 지연 로딩

#### 5.2 메모리 관리
```python
# 모델 사용 후 메모리 정리
shared_model.cleanup_memory()

# GPU 메모리 캐시 정리
torch.cuda.empty_cache()
gc.collect()
```

### 6. 성능 및 효율성 개선

#### 6.1 메모리 사용량
- 4비트 양자화로 메모리 사용량 약 4GB
- CPU 오프로드 지원으로 메모리 부족 시 자동 조정
- 모든 기능에서 동일한 모델 인스턴스 공유

#### 6.2 응답 속도
- 모델 로딩 시간 제거 (이미 메모리에 로드됨)
- HTTP 통신 오버헤드 제거 (별도 서비스 없음)
- 직접 모델 호출로 빠른 응답

### 7. 장점 및 특징

#### 7.1 아키텍처 장점
- **단순성**: 하나의 서버에서 모든 기능 처리
- **효율성**: 모델 중복 로드 없음
- **안정성**: 프로세스 간 통신 오류 가능성 없음
- **확장성**: RQ 워커를 통한 비동기 처리

#### 7.2 운영 장점
- **모니터링 용이**: 단일 서버에서 모든 로그 확인 가능
- **배포 간단**: 하나의 애플리케이션만 배포
- **리소스 효율**: 메모리 및 CPU 사용량 최적화

### 8. 테스트 방법

#### 8.1 전체 시스템 테스트
```bash
# 1. 모든 서비스 실행 확인
redis-cli ping  # PONG
curl http://localhost:8000/docs  # FastAPI 문서 접근 가능

# 2. 피드백 생성 테스트
curl -X POST http://localhost:8000/ai/feedback \
  -H "Content-Type: application/json" \
  -d @test_ai_feedback.json

# 3. RQ 워커에서 작업 처리 확인 (로그에서)
```

#### 8.2 모델 공유 확인
- 챗봇과 피드백 기능이 동일한 모델 사용
- 메모리 사용량이 일정하게 유지됨
- 모델 로딩 로그가 한 번만 출력됨

# 2025-07-01

## 주요 변경사항: 피드백 모델/워커 구조 점검 및 싱글톤 패턴 적용

### 1. FeedbackModel 싱글톤 패턴 적용
- FeedbackModel 클래스에 싱글톤 패턴을 적용하여, 한 프로세스 내에서 인스턴스가 한 번만 생성되도록 개선
- tasks.py에서 feedback_model = FeedbackModel()로 한 번만 인스턴스를 생성해 재사용
- generate_feedback_task 함수에서 매번 새로 인스턴스를 만들지 않도록 수정

### 2. shared_model 즉시 로딩 구조 확인
- shared_model(SharedMistralModel)은 이미 즉시 로딩 구조로, 프로세스가 시작될 때 모델이 메모리에 올라감
- FastAPI 서버와 RQ 워커는 각각 별도의 프로세스이므로, 각자 모델을 메모리에 올림(메모리 공유 불가)
- FastAPI 서버에서 로드한 모델을 RQ 워커가 직접 쓸 수 없음(파이썬 멀티프로세스 구조의 한계)

### 3. RQ 워커 구조 점검 및 실험
- 워커가 한 번만 실행되어 계속 살아있으면, 모델도 한 번만 로드됨(정상)
- 워커가 요청마다 새로 뜨거나, 프로세스가 죽고 다시 시작되면 매번 모델이 다시 로드됨(비정상)
- ps aux | grep "rq worker"로 워커 프로세스 상태 확인
- FeedbackModel, shared_model 생성 시점에 print/log 추가하여 실제 인스턴스 생성 횟수 실험

### 4. FastAPI와 RQ 워커의 모델 메모리 분리 원인
- FastAPI 서버와 RQ 워커는 완전히 다른 프로세스이기 때문에, 메모리(모델)를 공유할 수 없음
- 각 프로세스에서 shared_model을 import하면, 각자 자기 메모리에 모델을 올림
- 여러 워커를 띄우면 GPU 메모리도 그만큼 더 사용됨

### 5. 실무적 결론
- FastAPI와 RQ 워커 모두에서 shared_model이 한 번만 로드되는 구조가 정상
- 피드백 요청마다 모델이 다시 로드된다면, 워커가 죽거나, 코드 구조에 문제가 있을 가능성이 높음
- 멀티프로세스 환경에서 모델을 공유하려면 별도의 모델 서버(예: Triton, Ray Serve 등) 구조로 아키텍처를 변경해야 함

# 2025-07-03 vLLM 도입에 따른 피드백 모델 변경사항

## 1. vLLM 도입 배경 및 영향

### 피드백 모델 변경 필요성
- 챗봇 모델이 vLLM으로 전환되면서 일관성 있는 아키텍처 유지를 위해 피드백 모델도 vLLM 사용으로 변경
- 기존: Hugging Face Transformers 직접 사용 → 새로운: vLLM HTTP API 통신
- 메모리 사용량 최적화 및 성능 향상 목적

### vLLM 서버 공유
- 챗봇과 피드백 모델이 동일한 vLLM 서버(포트 8800)를 공유
- 모델 경로: `/home/ubuntu/mistral/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/e0bc86c23ce5aae1db576c8cca6f06f1f73af2db`
- 서버 리소스 효율성 향상

## 2. 코드 구조 변경사항

### 2.1. 기존 코드 (Hugging Face Transformers 직접 사용)
```python
# LLM_feedback_model.py
from Text.LLM.model.chatbot.shared_model import shared_model

class FeedbackModel:
    def __init__(self):
        # 공유 모델 사용
        self.model = shared_model.model
        self.tokenizer = shared_model.tokenizer
        
    async def generate_feedback(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # 메모리 정리
            shared_model.cleanup_memory()
            
            # 토크나이저로 입력 준비
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            # 모델 생성 설정
            generation_kwargs = dict(
                inputs,
                max_new_tokens=1024,
                temperature=0.7,
                do_sample=True,
                # ... 기타 설정
            )
            
            # 모델 추론
            outputs = self.model.generate(**generation_kwargs)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # ... 응답 처리
```

### 2.2. 새로운 코드 (vLLM HTTP API 사용)
```python
# LLM_feedback_model.py
import httpx

class FeedbackModel:
    def __init__(self):
        # vLLM 서버 URL 설정
        self.vllm_url = "http://localhost:8800/v1/chat/completions"
        self.model_path = "/home/ubuntu/mistral/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/e0bc86c23ce5aae1db576c8cca6f06f1f73af2db"
        
    async def generate_feedback(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # vLLM 서버에 POST 요청
            payload = {
                "model": self.model_path,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False  # 피드백은 스트리밍 불필요
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.vllm_url,
                    json=payload,
                    timeout=60.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    feedback_text = result["choices"][0]["message"]["content"]
                    # ... 응답 처리
```

## 3. 주요 변경사항

### 3.1. 모델 로딩 제거
- **기존**: `shared_model.model`, `shared_model.tokenizer` 직접 사용
- **변경**: vLLM 서버가 모델을 관리하므로 로컬 모델 로딩 불필요
- **효과**: RQ 워커의 메모리 사용량 대폭 감소

### 3.2. HTTP API 통신 방식
- **기존**: Python에서 직접 모델 추론
- **변경**: httpx를 사용한 HTTP API 통신
- **효과**: 더 안정적인 추론, 네트워크 기반 분산 처리 가능

### 3.3. 스트리밍 처리 제거
- **기존**: TextIteratorStreamer를 사용한 스트리밍 응답
- **변경**: 피드백은 완성된 텍스트만 필요하므로 스트리밍 불필요
- **효과**: 응답 처리 로직 단순화

### 3.4. 메모리 관리 단순화
- **기존**: 복잡한 메모리 정리 로직 (torch.cuda.empty_cache(), gc.collect())
- **변경**: vLLM 서버가 메모리 관리하므로 로컬 메모리 정리 불필요
- **효과**: 코드 단순화, 메모리 관리 오버헤드 제거

## 4. 성능 개선 효과

### 4.1. 메모리 사용량
- **기존**: RQ 워커에서 4GB (로컬 모델 로딩)
- **변경**: RQ 워커에서 ~0GB (vLLM 서버가 관리)
- **절약**: 100% 메모리 절약 (워커 기준)

### 4.2. 응답 속도
- **기존**: 모델 로딩 시간 + 추론 시간
- **변경**: HTTP 통신 시간 + 추론 시간
- **개선**: 모델 로딩 오버헤드 제거로 응답 속도 향상

### 4.3. 안정성
- **기존**: 토큰 디코딩 오버플로우 에러, 메모리 부족 문제
- **변경**: vLLM의 최적화된 추론 엔진으로 안정성 향상
- **개선**: 에러 발생 빈도 대폭 감소

### 4.4. 확장성
- **기존**: 단일 프로세스에서 모델 관리
- **변경**: vLLM 서버의 멀티프로세스/멀티스레드 지원
- **개선**: 동시 요청 처리 능력 향상

## 5. 아키텍처 변화

### 5.1. 기존 아키텍처
```
FastAPI 서버 (포트 8000)
    ↓
shared_model (4비트 양자화된 Mistral)
    ↓
├── 챗봇 기능 (base-info, free-text)
└── 피드백 기능 (feedback generation)

RQ 워커
    ↓
shared_model (4비트 양자화된 Mistral) - 별도 로딩
    ↓
피드백 생성
```

### 5.2. 새로운 아키텍처
```
FastAPI 서버 (포트 8000)
    ↓
vLLM 서버 (포트 8800) - 공유 모델
    ↓
├── 챗봇 기능 (HTTP API 호출)
└── 피드백 기능 (HTTP API 호출)

RQ 워커
    ↓
vLLM 서버 (포트 8800) - HTTP API 호출
    ↓
피드백 생성
```

## 6. 주의사항 및 고려사항

### 6.1. 의존성 추가
```python
# 새로운 의존성
import httpx  # HTTP 클라이언트
```

### 6.2. 서버 관리
- vLLM 서버가 챗봇과 피드백 모두에서 사용되므로 안정성 중요
- vLLM 서버 다운 시 전체 AI 기능 중단 가능성
- 모니터링 및 자동 재시작 스크립트 필요

### 6.3. 네트워크 의존성
- 로컬 HTTP 통신이므로 네트워크 지연 최소화
- vLLM 서버의 응답 시간이 전체 피드백 생성 시간에 영향

### 6.4. 에러 처리
- HTTP 통신 실패 시 적절한 에러 처리 필요
- vLLM 서버 응답 형식 변경 시 코드 수정 필요

## 7. 테스트 방법

### 7.1. vLLM 서버 상태 확인
```bash
# vLLM 서버 실행 확인
curl http://localhost:8800/v1/models

# 피드백 생성 테스트
curl -X POST http://localhost:8000/ai/feedback \
  -H "Content-Type: application/json" \
  -d @test_ai_feedback.json
```

### 7.2. RQ 워커 로그 확인
```bash
# 워커 실행
rq worker feedback

# 로그에서 vLLM 호출 확인
# "vLLM 호출" 로그 메시지 확인
```

## 8. 현재 상태
- 피드백 모델 vLLM 전환 완료
- 챗봇과 피드백 모델이 동일한 vLLM 서버 공유
- 메모리 사용량 최적화
- 안정성 향상
- 코드 구조 단순화
