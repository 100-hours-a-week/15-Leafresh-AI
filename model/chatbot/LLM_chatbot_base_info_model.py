# LLM_chatbot_base_info_model.py
from google.cloud import aiplatform
from vertexai.preview.generative_models import GenerativeModel
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from google.oauth2 import service_account
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from langfuse_config import langfuse_config
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
    ResponseSchema(name="recommend", description=f"추천 텍스트를 한 문장으로 출력해줘.(예: '이런 챌린지를 추천합니다.')"),
    ResponseSchema(name="challenges", description="추천 챌린지 리스트, 각 항목은 title, description 포함, description은 한 문장으로 요약해주세요.")
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

응답은 반드시 위 JSON 형식 그대로 출력하세요.

"""
)

# base-info_Output Parser 정의
def get_llm_response(prompt):
    try:
        trace_id = langfuse_config.create_trace_id()
        model = GenerativeModel(model_name=MODEL_NAME)
        response = model.generate_content(prompt)

        raw_text = response.text if hasattr(response, 'text') else response
    
        if isinstance(raw_text, dict): # dict이면 그대로 사용
            parsed = raw_text
        else:
            text = raw_text.strip()
            parsed = base_parser.parse(text)
            if isinstance(parsed.get("challenges"), str):
                parsed["challenges"] = json.loads(parsed["challenges"])
        return parsed

    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"챌린지 추천 중 내부 오류 발생: {str(e)}")
