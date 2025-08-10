# LLM_chatbot_base_info_model.py
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os
import json

import google.generativeai as genai

load_dotenv()

# Gemini 환경 변수 및 모델 초기화
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY_MAC")
MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY_MAC 환경 변수가 설정되어 있지 않습니다.")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

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
        response = model.generate_content(prompt)
        text = (response.text if hasattr(response, "text") else str(response)).strip()
        parsed = base_parser.parse(text)
        if isinstance(parsed.get("challenges"), str):
            parsed["challenges"] = json.loads(parsed["challenges"])
        return parsed
    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"챌린지 추천 중 내부 오류 발생: {str(e)}")
