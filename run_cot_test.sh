#!/bin/bash

echo "🚀 CoT(Chain of Thought) 테스트 실행 스크립트"
echo "=========================================="

# 환경 변수 확인
if [ -z "$LANGFUSE_PUBLIC_KEY" ] || [ -z "$LANGFUSE_SECRET_KEY" ]; then
    echo "❌ Langfuse 환경 변수가 설정되지 않았습니다."
    echo "다음 환경 변수를 설정해주세요:"
    echo "  - LANGFUSE_PUBLIC_KEY"
    echo "  - LANGFUSE_SECRET_KEY"
    echo ""
    echo "또는 .env 파일에 추가해주세요."
    exit 1
fi

# Python 가상환경 확인
if [ ! -d "venv" ] && [ ! -d ".venv" ]; then
    echo "📦 Python 가상환경을 생성합니다..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    if [ -d "venv" ]; then
        source venv/bin/activate
    else
        source .venv/bin/activate
    fi
fi

# 서버가 실행 중인지 확인
echo "🔍 서버 상태를 확인합니다..."
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ 서버가 실행 중입니다."
else
    echo "⚠️  서버가 실행되지 않았습니다. 서버를 먼저 실행해주세요:"
    echo "   python main.py"
    echo ""
    echo "또는 백그라운드에서 서버를 실행하시겠습니까? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "🔄 서버를 백그라운드에서 실행합니다..."
        python main.py &
        SERVER_PID=$!
        echo "서버 PID: $SERVER_PID"
        sleep 5
    else
        exit 1
    fi
fi

# CoT 테스트 실행
echo ""
echo "🧪 CoT 테스트를 실행합니다..."
python cot_test_runner.py

# 백그라운드 서버 종료 (있는 경우)
if [ ! -z "$SERVER_PID" ]; then
    echo ""
    echo "🛑 백그라운드 서버를 종료합니다..."
    kill $SERVER_PID
fi

echo ""
echo "✅ 테스트 완료!"
echo "📋 결과는 cot_test_report.md 파일에서 확인할 수 있습니다."
echo "🔗 Langfuse 대시보드에서 상세한 분석을 확인하세요." 