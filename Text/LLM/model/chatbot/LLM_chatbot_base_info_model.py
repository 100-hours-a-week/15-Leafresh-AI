from google.cloud import aiplatform
from vertexai.preview.generative_models import GenerativeModel
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from google.oauth2 import service_account
from fastapi import HTTPException
from fastapi.responses import JSONResponse
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

응답은 반드시 위 JSON형식 그대로 출력하세요.

"""
)

# base-info_Output Parser 정의
def get_llm_response(prompt):
    try:
        # 이미 초기화된 model 사용
        response = model.generate_content(prompt)
        
        if not response or not hasattr(response, 'text'):
            raise ValueError("모델 응답이 유효하지 않습니다.")

        raw_text = response.text.strip()
        print(f"Raw model response: {raw_text}")  # 디버깅용 로그

        try:
            # JSON 형식인지 확인
            if raw_text.startswith('{') and raw_text.endswith('}'):
                parsed = json.loads(raw_text)
            else:
                # JSON이 아니면 parser 사용
                parsed = base_parser.parse(raw_text)
                
            # challenges가 문자열인 경우 JSON으로 파싱
            if isinstance(parsed.get("challenges"), str):
                try:
                    parsed["challenges"] = json.loads(parsed["challenges"])
                except json.JSONDecodeError:
                    raise ValueError("challenges 형식이 올바르지 않습니다.")

            return parsed

        except json.JSONDecodeError as json_err:
            print(f"JSON 파싱 오류: {str(json_err)}")
            raise ValueError("응답 형식이 올바르지 않습니다.")

    except ValueError as val_err:
        print(f"값 검증 오류: {str(val_err)}")
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as e:
        print(f"예상치 못한 오류: {str(e)}")
        raise HTTPException(status_code=502, detail="AI 서버로부터 응답을 받아오는 데 실패했습니다.")
