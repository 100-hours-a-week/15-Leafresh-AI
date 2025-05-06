# Gemini LLM + LangChain
from chatbot_llm_model_vertex import base_prompt, get_llm_response
from chatbot_langchain_rag_chain import qa_chain, retriever
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import requests
import json
import re

app = FastAPI()

# 키워드 및 비속어 필터링 리스트
ENV_KEYWORDS = [
    "환경", "지구", "에코", "제로웨이스트", "탄소", "분리수거", "플라스틱", "텀블러", "기후", "친환경",
    "일회용", "미세먼지", "재활용", "자원", "대중교통", "도보", "비건", "탄소중립", "그린", "에너지", "쓰레기"
]
BAD_WORDS = [
    "시발", "씨발", "fuck", "shit", "개새끼", "병신", "ㅅㅂ", "ㅄ", "ㅂㅅ","fuckyou", "asshole", "tlqkf","ㅈ"
]

class CategoryRequest(BaseModel):
    memberId: int
    location: str
    workType: str
    category: str

class FreeTextRequest(BaseModel):
    memberId: int
    location: str
    workType: str
    userMessage: str

# 비-RAG 방식 챌린지 추천
@app.post("/ai/chatbot/recommendation/base-info")
def select_category(req: CategoryRequest):
    # 필수 필드 검사
    if not req.location:
        raise HTTPException(status_code=400, detail="location은 필수입니다.")
    if not req.workType:
        raise HTTPException(status_code=400, detail="workType은 필수입니다.")
    if not req.category:
        raise HTTPException(status_code=400, detail="category는 필수입니다.")
    # 필드 값 검사
    prompt = base_prompt.format(
        location=req.location,
        workType=req.workType,
        category=req.category
    )
    try:
        response = get_llm_response(prompt)
        return response
    except HTTPException as http_err:
        raise http_err  # 내부 HTTPException은 그대로 전달
    except Exception as e:
        raise HTTPException(status_code=502, detail="AI 서버로부터 추천 결과를 받아오는 데 실패했습니다.")

# LangChain 기반 RAG 추천
@app.post("/ai/chatbot/recommendation/free-text")
def freetext_rag(req: FreeTextRequest):
    # 필수 필드 검사
    if not req.userMessage:
        raise HTTPException(status_code=400, detail="userMessage는 필수입니다.")
    if len(req.userMessage.strip()) < 5:
        raise HTTPException(status_code=422, detail="userMessage는 문자열이어야 하며, 최소 5자 이상의 문자열이어야 합니다.")
        
    message_lower = req.userMessage.lower()
    if not any(k in req.userMessage for k in ENV_KEYWORDS) or any(b in message_lower for b in BAD_WORDS):
        return {
            "status": 403,
            "message": "저는 친환경 챌린지를 추천해드리는 Leafresh 챗봇이에요! 환경 관련 질문을 해주시면 더 잘 도와드릴 수 있어요.",
            "data": None
        }
    # 필수 필드 검사
    try:
        docs = retriever.invoke(req.userMessage)
        print(f" 검색된 문서 수: {len(docs)}")
        for i, doc in enumerate(docs):
            print(f" [문서 {i+1}] {doc.page_content}")

        context_text = "\n".join([doc.page_content for doc in docs])

        # PromptTemplate의 input_variables에 맞춰 context와 query를 전달
        variables = {
            "context": context_text,
            "query": req.userMessage
        }
        
        # LLM 응답 결과
        rag_result = qa_chain.invoke(variables)

        raw_result = rag_result.get("text", "")
        match = re.search(r'{.*}', raw_result, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                if isinstance(parsed.get("challenges"), str):
                    parsed["challenges"] = json.loads(parsed["challenges"])
                return {
                    "status": 200,
                    "message": "성공!",
                    "data": parsed
                }
            except Exception as parse_err:
                raise HTTPException(status_code=500, detail=f"챌린지 추천 중 내부 오류가 발생했습니다. 잠시 후 다시 시도해주세요. JSON 파싱 오류: {str(parse_err)}")
    except HTTPException as http_err:
        raise http_err  # 내부 HTTPException을 먼저 처리
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"AI 서버로부터 추천 결과를 받아오는 데 실패했습니다.\n{str(e)}")



# 백엔드와 테스트용으로 FastAPI와 Spring 간의 통신을 위한 엔드포인트 추가
# FastAPI → Spring: GET 
@app.get("/fastapi/call-spring")
def call_spring_get():
    res = requests.get("http://localhost:8080/spring/hello")
    return {"from_spring": res.text}

# FastAPI → Spring: POST
@app.post("/fastapi/call-spring")
def call_spring_post():
    res = requests.post("http://localhost:8080/spring/echo", json={"message": "방가방가!"})
    return {"from_spring": res.text}

# Spring → FastAPI: GET 수신
@app.get("/fastapi/hello")
def receive_get():
    return "Hello from FastAPI!"

# Spring → FastAPI: POST 수신
@app.post("/fastapi/echo")
async def receive_post(request: Request):
    data = await request.json()
    return {"fastapi_received": data.get("message", "방가방가!")}