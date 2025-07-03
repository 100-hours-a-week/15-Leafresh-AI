from prometheus_client import (generate_latest, Counter, Gauge, Histogram, REGISTRY)
from prometheus_client import process_collector, platform_collector

# collector 등록은 이미 되어 있으면 생략
if not any(type(c).__name__ == "ProcessCollector" for c in REGISTRY._collector_to_names):
    process_collector.ProcessCollector()
if not any(type(c).__name__ == "PlatformCollector" for c in REGISTRY._collector_to_names):
    platform_collector.PlatformCollector()
   
# 모델 추론 시간 (Histogram): 추론 시간 분포 파악
model_inference_duration_seconds = Histogram(
    'ai_model_inference_duration_seconds', 'Duration of AI model inference in seconds',
    buckets=(0.001, 0.01, 0.1, 1.0, 5.0, 10.0, float('inf'))
)

# 현재 서비스 중인 모델 버전 (Gauge)
model_version = Gauge('ai_current_model_version', 'Current AI model version')
model_version.set(2.1) # 실제 사용 중인 AI 모델 버전으로 설정