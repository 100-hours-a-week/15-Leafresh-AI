#!/bin/bash

# GCP Pub/Sub 워커 시작 스크립트

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 가상환경 활성화
if [ -f ~/.venv/bin/activate ]; then
    source ~/.venv/bin/activate
    echo -e "${GREEN}가상환경 활성화됨: $(which python)${NC}"
else
    echo -e "${YELLOW}가상환경을 찾을 수 없습니다. 시스템 Python을 사용합니다.${NC}"
fi

# 로그 디렉토리 생성
LOG_DIR="./logs"
mkdir -p $LOG_DIR

# PID 파일 디렉토리
PID_DIR="./pids"
mkdir -p $PID_DIR

echo -e "${BLUE}=== GCP Pub/Sub 피드백 워커 시작 ===${NC}"

# 인자에 따라 환경 분기
if [ "$1" == "mac" ]; then
    export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/leafresh-mac-6256c2349946.json"
    export GOOGLE_CLOUD_PROJECT="leafresh-mac"
    echo "환경: leafresh-mac"
elif [ "$1" == "dev2" ]; then
    export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/leafresh-dev2-compute-sa.json"
    export GOOGLE_CLOUD_PROJECT="leafresh-dev2"
    echo "환경: leafresh-dev2"
else
    echo "사용법: $0 [mac|dev2]"
    exit 1
fi

# Pub/Sub 토픽/구독 환경변수 설정
if [ -z "$PUBSUB_TOPIC_FEEDBACK_DEV" ]; then
    export PUBSUB_TOPIC_FEEDBACK_DEV="leafresh-feedback-topic"
fi

if [ -z "$PUBSUB_SUBSCRIPTION_FEEDBACK_DEV" ]; then
    export PUBSUB_SUBSCRIPTION_FEEDBACK_DEV="leafresh-feedback-sub"
fi

if [ -z "$PUBSUB_TOPIC_FEEDBACK_RESULT_DEV" ]; then
    export PUBSUB_TOPIC_FEEDBACK_RESULT_DEV="leafresh-feedback-result-topic"
fi

if [ -z "$PUBSUB_TOPIC_FEEDBACK_DLQ_DEV" ]; then
    export PUBSUB_TOPIC_FEEDBACK_DLQ_DEV="leafresh-feedback-result-dlq-topic"
fi

echo -e "${GREEN}환경변수 설정 완료:${NC}"
echo -e "  GOOGLE_CLOUD_PROJECT: $GOOGLE_CLOUD_PROJECT"
echo -e "  PUBSUB_TOPIC_FEEDBACK_DEV: $PUBSUB_TOPIC_FEEDBACK_DEV"
echo -e "  PUBSUB_SUBSCRIPTION_FEEDBACK_DEV: $PUBSUB_SUBSCRIPTION_FEEDBACK_DEV"
echo -e "  PUBSUB_TOPIC_FEEDBACK_RESULT_DEV: $PUBSUB_TOPIC_FEEDBACK_RESULT_DEV"
echo -e "  PUBSUB_TOPIC_FEEDBACK_DLQ_DEV: $PUBSUB_TOPIC_FEEDBACK_DLQ_DEV"

# 기존 Pub/Sub 워커 프로세스 확인 및 종료
PUBSUB_PIDS=$(pgrep -f "gcp_pubsub_worker.py")
if [ ! -z "$PUBSUB_PIDS" ]; then
    echo -e "${YELLOW}기존 GCP Pub/Sub 워커 프로세스 종료 중...${NC}"
    for pid in $PUBSUB_PIDS; do
        kill $pid
        sleep 1
        if ps -p $pid > /dev/null 2>&1; then
            kill -9 $pid
        fi
    done
fi

# GCP Pub/Sub 워커 백그라운드 실행
echo -e "${YELLOW}GCP Pub/Sub 워커 시작 중...${NC}"
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)"
nohup python model/feedback/gcp_pubsub_worker.py > $LOG_DIR/gcp_pubsub_worker.log 2>&1 &

PUBSUB_PID=$!
echo $PUBSUB_PID > $PID_DIR/gcp_pubsub_worker.pid

echo -e "${GREEN}✓ GCP Pub/Sub 워커 시작 완료 (PID: $PUBSUB_PID)${NC}"
echo -e "${BLUE}로그 확인: tail -f $LOG_DIR/gcp_pubsub_worker.log${NC}"

# 토픽 구조 안내
echo -e "${BLUE}=== GCP Pub/Sub 토픽 구조 ===${NC}"
echo -e "${YELLOW}[Wren] ──▶ [Mac]:${NC}"
echo -e "  $PUBSUB_TOPIC_FEEDBACK_DEV ──▶ $PUBSUB_SUBSCRIPTION_FEEDBACK_DEV"
echo -e ""
echo -e "${YELLOW}[Mac] ──▶ [Wren]:${NC}"
echo -e "  $PUBSUB_TOPIC_FEEDBACK_RESULT_DEV"
echo -e "    └──▶ leafresh-feedback-result-sub (구독자: Wren)"
echo -e "          └──(DLQ)▶ $PUBSUB_TOPIC_FEEDBACK_DLQ_DEV ─▶ leafresh-feedback-result-dlq-sub"

echo -e "${BLUE}=== 워커 시작 완료 ===${NC}" 