#!/bin/bash

# 서비스 시작 스크립트
# Redis, vLLM, FastAPI 서버를 백그라운드로 실행

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

echo -e "${BLUE}=== Leafresh AI 서비스 시작 ===${NC}"

# 함수: 서비스 상태 확인
check_service() {
    local service_name=$1
    local port=$2
    local pid_file="$PID_DIR/${service_name}.pid"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p $pid > /dev/null 2>&1; then
            echo -e "${GREEN}✓ $service_name (PID: $pid) 실행 중${NC}"
            return 0
        else
            echo -e "${RED}✗ $service_name (PID: $pid) 중단됨${NC}"
            rm -f "$pid_file"
            return 1
        fi
    else
        echo -e "${YELLOW}? $service_name 상태 확인 불가${NC}"
        return 1
    fi
}

# 함수: 포트 사용 확인
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}


# 1. FastAPI 서버 시작
echo -e "${YELLOW}2. FastAPI 서버 시작 중...${NC}"
if check_port 8000; then
    echo -e "${GREEN}FastAPI 서버가 이미 실행 중입니다 (포트 8000)${NC}"
else
    # FastAPI 서버 백그라운드 실행
    cd "$(dirname "$0")"
    nohup python main.py > $LOG_DIR/fastapi.log 2>&1 &
    
    FASTAPI_PID=$!
    echo $FASTAPI_PID > $PID_DIR/fastapi.pid
    
    # FastAPI 서버 시작 대기
    echo -e "${YELLOW}FastAPI 서버 시작 대기 중... (최대 30초)${NC}"
    for i in {1..30}; do
        if check_port 8000; then
            echo -e "${GREEN}✓ FastAPI 서버 시작 완료 (PID: $FASTAPI_PID)${NC}"
            break
        fi
        if [ $i -eq 30 ]; then
            echo -e "${RED}✗ FastAPI 서버 시작 실패 (30초 타임아웃)${NC}"
            exit 1
        fi
        sleep 1
    done
fi

# 2. vLLM 서버 시작
echo -e "${YELLOW}4. vLLM 서버 시작 중...${NC}"
if check_port 8800; then
    echo -e "${GREEN}vLLM 서버가 이미 실행 중입니다 (포트 8800)${NC}"
else
    # vLLM 서버 백그라운드 실행 (KV 캐시 부족 문제 해결을 위한 파라미터 추가)
    nohup python -m vllm.entrypoints.openai.api_server \
        --model /home/ubuntu/mistral/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/e0bc86c23ce5aae1db576c8cca6f06f1f73af2db \
        --host 0.0.0.0 \
        --port 8800 \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.9
        # --tensor-parallel-size 1 \
        # --enforce-eager \
        > $LOG_DIR/vllm.log 2>&1 &
    
    VLLM_PID=$!
    echo $VLLM_PID > $PID_DIR/vllm.pid
    
    # vLLM 서버 시작 대기
    echo -e "${YELLOW}vLLM 서버 시작 대기 중... (최대 60초)${NC}"
    for i in {1..60}; do
        if check_port 8800; then
            echo -e "${GREEN}✓ vLLM 서버 시작 완료 (PID: $VLLM_PID)${NC}"
            break
        fi
        if [ $i -eq 60 ]; then
            echo -e "${RED}✗ vLLM 서버 시작 실패 (60초 타임아웃)${NC}"
            exit 1
        fi
        sleep 1
    done
fi

echo -e "${BLUE}=== 모든 서비스 시작 완료 ===${NC}"
echo -e "${GREEN}서비스 상태:${NC}"
check_service "redis" 6379
check_service "vllm" 8800
check_service "fastapi" 8000

# RQ 워커 상태 확인
echo -e "${YELLOW}RQ 워커 상태:${NC}"
for i in {1..1}; do
    check_service "rq_worker_${i}" 0
done

echo -e "${BLUE}로그 파일 위치:${NC}"
echo -e "  Redis: $LOG_DIR/redis.log"
echo -e "  vLLM: $LOG_DIR/vllm.log"
echo -e "  FastAPI: $LOG_DIR/fastapi.log"
echo -e "  RQ Worker 1: $LOG_DIR/rq_worker_1.log"

echo -e "${BLUE}PID 파일 위치:${NC}"
echo -e "  $PID_DIR/"

echo -e "${YELLOW}서비스 중지하려면: ./stop_services.sh${NC}"
echo -e "${YELLOW}서비스 상태 확인: ./status_services.sh${NC}" 