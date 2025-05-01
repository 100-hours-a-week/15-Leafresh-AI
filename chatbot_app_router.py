# chatbot_app_router.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from chatbot_llm_model_vertex import get_llm_response
from chatbot_langchain_rag_chain import qa_chain
import json

# 키워드 및 비속어 필터링 리스트
ENV_KEYWORDS = [
        "환경", "지구", "에코", "제로웨이스트", "탄소", "분리수거", "플라스틱", "텀블러", "기후", "친환경",
        "일회용", "미세먼지", "재활용", "자원", "대중교통", "도보", "비건", "탄소중립", "그린", "에너지", "쓰레기"
    ]
BAD_WORDS = [
        "시발", "씨발", "좆", "fuck", "shit", "개새끼", "병신", "ㅅㅂ", "ㅄ", "ㅂㅅ",
        "fuckyou", "asshole", "tlqkf", "좃", "개"
    ]

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello from Gemini!"}

class CategoryRequest(BaseModel):
    userId: int
    category: str
    location: str
    work_type: str

class FreeTextRequest(BaseModel):
    userId: int
    location: str
    work_type: str
    user_message: str

@app.post("/ai/chatbot/recommendation/base-info")
def select_category(req: CategoryRequest):
    # 비-RAG 방식: LLM 기반 기본 챌린지 추천
    prompt = f"""
    {req.location} 환경에 있는 {req.work_type} 사용자가 {req.category}를 실천할 때,
    절대적으로 환경에 도움이 되는 챌린지를 아래 JSON 형식으로 3가지 추천해줘:
    반드시 순수 JSON만 출력해줘.
    {{
        "status": 200,
        "message": "성공!",
        "data": {{
            "recommand": "설명 텍스트",
            "challenges": [
                {{"title": "챌린지 이름", "description": "설명"}}
            ]
        }}
    }}
    """
    return get_llm_response(prompt)

@app.post("/chatbot/freetext")
def free_text(req: FreeTextRequest):
    if not req.user_message.strip():
        raise HTTPException(status_code=400, detail="user_message는 필수입니다.")

    if not isinstance(req.user_message, str) or len(req.user_message.strip()) < 5:
        raise HTTPException(status_code=422, detail="user_message는 문자열이어야 하며, 최소한의 의미를 가져야 합니다.")

    prompt = f"""
    사용자 메시지: "{req.user_message}"
    지역: {req.location}, 직업: {req.work_type}
    환경에 도움이 되는 챌린지를 아래 JSON 형식으로만 3가지 추천해주세요:
    {{
        "status": 200,
        "message": "성공!",
        "data": {{
            "recommand": "설명 텍스트",
            "challenges": [
                {{"title": "챌린지 이름", "description": "설명"}}
            ]
        }}
    }}
    """
    return get_llm_response(prompt)

@app.post("/ai/chatbot/recommendation/free-text")
def freetext_rag(req: FreeTextRequest):

    # 기본 입력 검증
    if req.user_message is None or not req.user_message.strip():
        raise HTTPException(status_code=400, detail="user_message는 필수입니다.")

    if not isinstance(req.user_message, str) or len(req.user_message.strip()) < 5:
        raise HTTPException(status_code=422, detail="user_message는 문자열이어야 하며, 최소한 5자 이상이어야 합니다.")

    # 비속어 또는 친환경 주제 미포함 필터링
    message_lower = req.user_message.lower()  # 소문자 변환 후 검사 (영어 욕설 대응)

    if not any(keyword in req.user_message for keyword in ENV_KEYWORDS) or \
       any(bad in message_lower for bad in BAD_WORDS):
        return {
            "status": 403,
            "message": "저는 친환경 챌린지를 추천해드리는 Leafresh 챗봇이에요! 환경 관련 질문을 해주시면 더 잘 도와드릴 수 있어요.",
            "data": None
        }

    # LangChain 기반 RAG 체인 호출
    try:
        result = qa_chain.run(req.user_message)
        print("🔍 LLM 응답 원문:\n", result)
        # RAG 응답 파싱
        parsed = json.loads(result)
        return parsed

    except json.JSONDecodeError:
        return {
            "status": 500,
            "message": "❌ RAG 응답에서 JSON 파싱 실패",
            "data": None
        }
    except Exception as e:
        return {
            "status": 500,
            "message": f"❌ RAG 호출 중 오류 발생: {str(e)}",
            "data": None
        }