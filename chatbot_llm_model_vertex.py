# chatbot_llm_model_vertex.py
# /ai/chatbot/recommendation/base-info 에서 받아와 답변 호출
from google.cloud import aiplatform
from vertexai.preview.generative_models import GenerativeModel
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from google.oauth2 import service_account
from fastapi import HTTPException
from dotenv import load_dotenv
import os
import json

load_dotenv()

#  환경 변수에서 값 가져오기
SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("VERTEX_AI_LOCATION")
MODEL_NAME = os.getenv("VERTEX_MODEL_NAME")

# Vertex AI 초기화
credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
aiplatform.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)

# GenerativeModel 초기화(SDK 방식 사용)
model = GenerativeModel(model_name=MODEL_NAME)

# base-info_response_schemas 정의
base_response_schemas = [
    ResponseSchema(name="recommend", description="추천 설명 텍스트"),
    ResponseSchema(name="challenges", description="추천 챌린지 리스트, 각 항목은 title, description 포함")
]

# base-info_output_parser 정의 
base_parser = StructuredOutputParser.from_response_schemas(base_response_schemas)

# base-info_prompt 정의
escaped_format = base_parser.get_format_instructions().replace("{", "{{").replace("}", "}}")
base_prompt = PromptTemplate(
    input_variables=["location", "workType", "category"],
    template=f"""
{{location}} 환경에 있는 {{workType}} 사용자가 {{category}}를 실천할 때,
절대적으로 환경에 도움이 되는 챌린지를 아래 JSON 형식으로 3가지 추천해주세요.

JSON 포맷:
{escaped_format}

응답은 반드시 위 JSON 형식 그대로, 마크다운 없이 순수 JSON만 출력하세요.
"""
)

# base-info_Output Parser 정의
def get_llm_response(prompt):
    try:
        model = GenerativeModel(model_name=MODEL_NAME)
        response = model.generate_content(prompt)
        raw_text = response.text if hasattr(response, 'text') else response
        if isinstance(raw_text, dict):
            raw_text = str(raw_text)

        if not isinstance(raw_text, str):
            raise ValueError("LLM 응답이 문자열이 아닙니다.")

        text = raw_text.strip()
        parsed = base_parser.parse(text)
        if isinstance(parsed.get("challenges"), str):
            parsed["challenges"] = json.loads(parsed["challenges"])  # JSON 문자열 파싱
        return {
            "status": 200,
            "message": "성공!",
            "data": parsed
        }
    except HTTPException as http_err:
        raise http_err  # 내부 HTTPException은 그대로 전달
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"챌린지 추천 중 내부 오류가 발생했습니다. 잠시 후 다시 시도해주세요.\n{str(e)}")
