from prometheus_client import (generate_latest, Counter, Gauge, Histogram, REGISTRY, process_collector, platform_collector)
from fastapi import Request
import time

# collector 등록은 이미 되어 있으면 생략
if not any(type(c).__name__ == "ProcessCollector" for c in REGISTRY._collector_to_names):
    process_collector.ProcessCollector()
if not any(type(c).__name__ == "PlatformCollector" for c in REGISTRY._collector_to_names):
    platform_collector.PlatformCollector()

# API 요청 수 카운터 
api_requests_total = Counter(
    'api_requests_total', 'Total number of API requests',
    ['endpoint', 'method', 'status_code']
)
   
# 모델 추론 시간 (Histogram): 추론 시간 분포 파악
http_request_duration_seconds = Histogram(
    'http_request_duration_seconds', 'Duration of HTTP requests in seconds',
    ['endpoint', 'method'],
    buckets=(0.001, 0.01, 0.1, 1.0, 5.0, 10.0, float('inf'))
)

# 현재 서비스 중인 모델 버전 (Gauge)
model_version = Gauge('ai_current_model_version', 'Current AI model version')
model_version.set(2.1) # 실제 사용 중인 AI 모델 버전으로 설정

async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    http_request_duration_seconds.labels(
        endpoint=request.url.path,
        method=request.method
    ).observe(duration)

    # 요청 수 카운트
    api_requests_total.labels(
        endpoint=request.url.path,
        method=request.method,
        status_code=str(response.status_code)
    ).inc()

    return response