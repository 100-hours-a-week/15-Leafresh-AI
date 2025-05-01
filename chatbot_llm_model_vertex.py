# chatbot_llm_model_vertex.py
import vertexai
from vertexai.generative_models import GenerativeModel
from google.oauth2 import service_account
from dotenv import load_dotenv
import os
import json
import re

load_dotenv()

#  환경 변수에서 값 가져오기
SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "kakao-project-457106")
LOCATION = os.getenv("VERTEX_AI_LOCATION", "us-central1")
MODEL_NAME = os.getenv("VERTEX_MODEL_NAME", "publishers/google/models/gemini-1.5-flash")

# Vertex AI 초기화
credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)

#  모델 객체 생성
model = GenerativeModel("gemini-1.5-flash")

#  LLM 응답 함수
def get_llm_response(prompt: str):
    try:
        response = model.generate_content(prompt)

        # 안전하게 텍스트 추출
        raw_text = getattr(response, "text", "")
        print("📦 Gemini 응답 전체:", raw_text)

        # JSON 형식 추출
        match = re.search(r'{.*}', raw_text, re.DOTALL)
        if match:
            return json.loads(match.group())

        return {"status": 500, "message": "❌ JSON 블록 파싱 실패", "data": None}

    except Exception as e:
        return {"status": 500, "message": f"❌ Gemini 호출 실패: {str(e)}", "data": None}