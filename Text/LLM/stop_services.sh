#!/bin/bash

# 서비스 중지 스크립트
# Redis, vLLM, FastAPI 서버를 안전하게 중지

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# PID 파일 디렉토리
PID_DIR="./pids"

echo -e "${BLUE}=== Leafresh AI 서비스 중지 ===${NC}"

# 함수: 프로세스 안전 종료
kill_service() {
    local service_name=$1
    local pid_file="$PID_DIR/${service_name}.pid"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p $pid > /dev/null 2>&1; then
            echo -e "${YELLOW}$service_name (PID: $pid) 종료 중...${NC}"
            kill $pid
            
            # 프로세스 종료 대기 (최대 10초)
            for i in {1..10}; do
                if ! ps -p $pid > /dev/null 2>&1; then
                    echo -e "${GREEN}✓ $service_name 종료 완료${NC}"
                    rm -f "$pid_file"
                    return 0
                fi
                sleep 1
            done
            
            # 강제 종료
            echo -e "${YELLOW}$service_name 강제 종료 중...${NC}"
            kill -9 $pid
            sleep 1
            if ! ps -p $pid > /dev/null 2>&1; then
                echo -e "${GREEN}✓ $service_name 강제 종료 완료${NC}"
                rm -f "$pid_file"
            else
                echo -e "${RED}✗ $service_name 종료 실패${NC}"
                return 1
            fi
        else
            echo -e "${YELLOW}$service_name (PID: $pid) 이미 종료됨${NC}"
            rm -f "$pid_file"
        fi
    else
        echo -e "${YELLOW}$service_name PID 파일 없음${NC}"
    fi
}

# 1. FastAPI 서버 중지
echo -e "${YELLOW}1. FastAPI 서버 중지 중...${NC}"
kill_service "fastapi"

# 2. vLLM 서버 중지
echo -e "${YELLOW}2. vLLM 서버 중지 중...${NC}"
kill_service "vllm"

# 3. RQ 워커 중지
echo -e "${YELLOW}3. RQ 워커 중지 중...${NC}"
for i in {1..2}; do
    kill_service "rq_worker_${i}"
done

# RQ 워커 프로세스 추가 종료 시도
RQ_PIDS=$(pgrep -f "rq worker feedback")
if [ ! -z "$RQ_PIDS" ]; then
    echo -e "${YELLOW}RQ 워커 프로세스 추가 종료 중...${NC}"
    for pid in $RQ_PIDS; do
        echo -e "${YELLOW}RQ 워커 프로세스 (PID: $pid) 종료 중...${NC}"
        kill $pid
        sleep 1
        if ps -p $pid > /dev/null 2>&1; then
            kill -9 $pid
            echo -e "${GREEN}RQ 워커 프로세스 강제 종료 완료${NC}"
        fi
    done
fi

# 4. Redis 서버 중지
echo -e "${YELLOW}4. Redis 서버 중지 중...${NC}"
kill_service "redis"

# Redis 서버 추가 종료 시도 (redis-server 프로세스)
REDIS_PIDS=$(pgrep redis-server)
if [ ! -z "$REDIS_PIDS" ]; then
    echo -e "${YELLOW}Redis 서버 프로세스 추가 종료 중...${NC}"
    for pid in $REDIS_PIDS; do
        echo -e "${YELLOW}Redis 프로세스 (PID: $pid) 종료 중...${NC}"
        kill $pid
        sleep 1
        if ps -p $pid > /dev/null 2>&1; then
            kill -9 $pid
            echo -e "${GREEN}Redis 프로세스 강제 종료 완료${NC}"
        fi
    done
fi

echo -e "${BLUE}=== 모든 서비스 중지 완료 ===${NC}"

# 포트 사용 확인
echo -e "${BLUE}포트 사용 상태:${NC}"
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${RED}포트 8000 (FastAPI) 여전히 사용 중${NC}"
else
    echo -e "${GREEN}포트 8000 (FastAPI) 사용 안함${NC}"
fi

if lsof -Pi :8800 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${RED}포트 8800 (vLLM) 여전히 사용 중${NC}"
else
    echo -e "${GREEN}포트 8800 (vLLM) 사용 안함${NC}"
fi

if lsof -Pi :6379 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${RED}포트 6379 (Redis) 여전히 사용 중${NC}"
else
    echo -e "${GREEN}포트 6379 (Redis) 사용 안함${NC}"
fi 