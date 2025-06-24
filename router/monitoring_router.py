import time
from fastapi import APIRouter
from starlette.responses import PlainTextResponse
from starlette.requests import Request  # 통일

from prometheus_client import (
    generate_latest, Counter, Gauge, Histogram, collect_default_metrics
)

collect_default_metrics() # Python 프로세스 기본 지표 수집
   
# API 요청 카운터: 엔드포인트, 메소드, 상태 코드를 라벨로 사용
api_requests_total = Counter(
    'ai_api_requests_total', 'Total AI API requests', ['endpoint', 'method', 'status_code']
)

# 모델 추론 시간 (Histogram): 추론 시간 분포 파악
model_inference_duration_seconds = Histogram(
    'ai_model_inference_duration_seconds', 'Duration of AI model inference in seconds',
    buckets=(0.001, 0.01, 0.1, 1.0, 5.0, 10.0, float('inf'))
)

# 현재 활성 요청 수 (Gauge)
active_requests = Gauge('ai_active_requests', 'Number of active AI requests')

# 현재 서비스 중인 모델 버전 (Gauge)
model_version = Gauge('ai_current_model_version', 'Current AI model version')
model_version.set(2.1) # 실제 사용 중인 AI 모델 버전으로 설정

# 라우터 설정 
router = APIRouter()

@router.get("/metrics")
async def metrics():
    # 모든 메트릭을 Prometheus 형식으로 반환
    return PlainTextResponse(generate_latest().decode('utf-8'))

async def add_process_time_header(request: Request, call_next):
    active_requests.inc() # 요청 시작 시 활성 요청 수 증가
    start_time = time.time()

    response = await call_next(request) # 다음 미들웨어/라우트 핸들러 호출

    process_time = time.time() - start_time
    model_inference_duration_seconds.observe(process_time) # 전체 요청 처리 시간 기록

    # API 요청 카운터 기록
    endpoint_path = request.url.path
    method = request.method
    status_code = response.status_code

    api_requests_total.labels(
        endpoint=endpoint_path,
        method=method,
        status_code=status_code
    ).inc()

    active_requests.dec() # 요청 완료 시 활성 요청 수 감소
    return response