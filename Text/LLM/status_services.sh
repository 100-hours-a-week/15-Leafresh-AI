#!/bin/bash

# 서비스 상태 확인 스크립트
# Redis, vLLM, FastAPI 서버의 실행 상태를 확인

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# PID 파일 디렉토리
PID_DIR="./pids"
LOG_DIR="./logs"

echo -e "${BLUE}=== Leafresh AI 서비스 상태 ===${NC}"

# 함수: 포트 사용 확인
check_port() {
    local port=$1
    local service_name=$2
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        local pid=$(lsof -Pi :$port -sTCP:LISTEN -t)
        echo -e "${GREEN}✓ $service_name (포트 $port) 실행 중 (PID: $pid)${NC}"
        return 0
    else
        echo -e "${RED}✗ $service_name (포트 $port) 중단됨${NC}"
        return 1
    fi
}

# 함수: PID 파일 기반 상태 확인
check_pid_file() {
    local service_name=$1
    local pid_file="$PID_DIR/${service_name}.pid"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p $pid > /dev/null 2>&1; then
            echo -e "${GREEN}✓ $service_name PID 파일 확인됨 (PID: $pid)${NC}"
            return 0
        else
            echo -e "${RED}✗ $service_name PID 파일 있지만 프로세스 없음 (PID: $pid)${NC}"
            return 1
        fi
    else
        echo -e "${YELLOW}? $service_name PID 파일 없음${NC}"
        return 1
    fi
}

# 함수: 로그 파일 크기 확인
check_log_size() {
    local service_name=$1
    local log_file="$LOG_DIR/${service_name}.log"
    
    if [ -f "$log_file" ]; then
        local size=$(du -h "$log_file" | cut -f1)
        local lines=$(wc -l < "$log_file")
        echo -e "${BLUE}  로그 파일: $log_file (크기: $size, 라인: $lines)${NC}"
    else
        echo -e "${YELLOW}  로그 파일 없음: $log_file${NC}"
    fi
}

# 1. Redis 서버 상태
echo -e "${YELLOW}1. Redis 서버:${NC}"
check_port 6379 "Redis"
check_pid_file "redis"
check_log_size "redis"

# 2. vLLM 서버 상태
echo -e "${YELLOW}2. vLLM 서버:${NC}"
check_port 8800 "vLLM"
check_pid_file "vllm"
check_log_size "vllm"

# 3. FastAPI 서버 상태
echo -e "${YELLOW}3. FastAPI 서버:${NC}"
check_port 8000 "FastAPI"
check_pid_file "fastapi"
check_log_size "fastapi"

# 4. RQ 워커 상태
echo -e "${YELLOW}4. RQ 워커:${NC}"
for i in {1..2}; do
    echo -e "${BLUE}  워커 ${i}:${NC}"
    check_pid_file "rq_worker_${i}"
    check_log_size "rq_worker_${i}"
done

# RQ 워커 프로세스 확인
RQ_PIDS=$(pgrep -f "rq worker feedback")
if [ ! -z "$RQ_PIDS" ]; then
    echo -e "${GREEN}  실행 중인 RQ 워커 프로세스:${NC}"
    for pid in $RQ_PIDS; do
        echo -e "    PID: $pid"
    done
else
    echo -e "${RED}  실행 중인 RQ 워커 프로세스 없음${NC}"
fi

# 5. 시스템 리소스 사용량
echo -e "${YELLOW}5. 시스템 리소스:${NC}"
echo -e "${BLUE}  CPU 사용률:${NC}"
top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1

echo -e "${BLUE}  메모리 사용률:${NC}"
free -h | grep "Mem:" | awk '{print "사용: " $3 "/" $2 " (" int($3/$2*100) "%)"}'

echo -e "${BLUE}  GPU 사용률:${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | while IFS=, read -r gpu_util mem_used mem_total; do
        echo -e "  GPU 0: ${gpu_util}% 사용률, ${mem_used}MiB/${mem_total}MiB 메모리"
    done
else
    echo -e "${YELLOW}  nvidia-smi 명령어 없음${NC}"
fi

# 6. 최근 로그 확인
echo -e "${YELLOW}6. 최근 로그 (마지막 5줄):${NC}"
for service in redis vllm fastapi; do
    log_file="$LOG_DIR/${service}.log"
    if [ -f "$log_file" ]; then
        echo -e "${BLUE}  $service 로그:${NC}"
        tail -5 "$log_file" | sed 's/^/    /'
    fi
done

# RQ 워커 로그 확인
for i in {1..2}; do
    log_file="$LOG_DIR/rq_worker_${i}.log"
    if [ -f "$log_file" ]; then
        echo -e "${BLUE}  RQ 워커 ${i} 로그:${NC}"
        tail -5 "$log_file" | sed 's/^/    /'
    fi
done

echo -e "${BLUE}=== 상태 확인 완료 ===${NC}" 