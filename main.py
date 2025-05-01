# main.py
import os
from dotenv import load_dotenv
load_dotenv()

import nest_asyncio
from pyngrok import ngrok
import uvicorn

# ✅ ngrok 초기화
ngrok.kill()
nest_asyncio.apply()

load_dotenv() # .env 파일에서 환경 변수 로드

# ✅ 환경 변수에서 ngrok 토큰 불러오기 (권장)
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN")
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# ✅ 포트 연결 및 공개 주소 획득
public_url = ngrok.connect(8000)
print(f"🚀 서버 실행 주소: {public_url}")

# ✅ uvicorn으로 FastAPI 앱 실행
uvicorn.run("chatbot_app_router:app", host="0.0.0.0", port=8000)