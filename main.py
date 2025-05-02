# main.py
import os
from dotenv import load_dotenv
import uvicorn

load_dotenv()

# uvicorn으로 FastAPI 앱 실행
print("🚀 FastAPI 서버가 http://0.0.0.0:8000 에서 실행됩니다.")
uvicorn.run("chatbot_app_router:app", host="0.0.0.0", port=8000)