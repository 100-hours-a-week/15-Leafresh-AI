from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_qdrant import Qdrant
from langchain_community.embeddings import SentenceTransformerEmbeddings
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.memory import ConversationBufferMemory
from typing import Dict, Generator, Any, List, Optional, TypedDict, Annotated, Sequence
from Text.LLM.model.chatbot.chatbot_constants import label_mapping, category_keywords
import os
import json
import re
import logging
import httpx
import unicodedata
import random

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

# Qdrant Cloud URL을 사용하여 클라이언트를 초기화하도록 수정
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

embeddings = SentenceTransformerEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device': 'cpu'}
)

# Qdrant DB 연결
qdrant = Qdrant(
    client=qdrant_client,
    collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
    embeddings=embeddings,
)
retriever = qdrant.as_retriever(search_kwargs={"k": 3})

logger.info("Qdrant DB for free-text chatbot connected successfully.")

class ChatState(TypedDict):
    messages: Annotated[Sequence[Dict[str, str]], "대화 기록"]
    current_query: str
    context: str
    response: str
    should_continue: bool
    error: Optional[str]
    docs: Optional[list]
    sessionId: str
    category: Optional[str]
    base_category: Optional[str]

conversation_states: Dict[str, ChatState] = {}

free_text_response_schemas = [
    ResponseSchema(name="recommend", description="추천 텍스트를 한글로 한 문장으로 출력해 주세요. (예: '이런 챌린지를 추천합니다.')"),
    ResponseSchema(name="challenges", description="추천 챌린지 리스트, 각 항목은 title, description 포함, description은 한글로 한 문장으로 요약해주세요.")
]

free_text_parser = StructuredOutputParser.from_response_schemas(free_text_response_schemas)

# [변경사항 주석] 2024-07-31
# 프롬프트 변수명을 custom_prompt로 지정
custom_prompt = PromptTemplate(
    input_variables=["context", "query", "messages", "category"],
    template="""너는 사용자의 일상 대화를 이해하고, 대화의 맥락에 맞춰 친환경 챌린지를 추천하는 챗봇이야.

[대화 기록]
{messages}

[관련 정보]
{context}

[사용자 현재 입력]
{query}

[관심 카테고리]
{category}

[중요 요구사항]
- 대화의 흐름과 관련 정보를 종합하여, 사용자가 흥미를 느낄 만한 친환경 챌린지 3가지를 추천해줘.
- 반드시 아래 예시와 같은 마크다운과 JSON 형식으로만 응답해야 해. 다른 설명은 절대 추가하지 마.
- 모든 내용은 반드시 한글로 작성하고, 문장 끝은 "니다." 또는 "요."로 자연스럽게 끝내줘.
- 각 챌린지는 "title"과 "description" 필드만 포함해야 해.
- "title"은 "1. ", "2. ", "3. " 형식으로 번호를 붙여서 시작해.
- "description"은 한 문장으로 간결하게 요약해 줘.
- 영어, 이모티콘, 불필요한 특수문자는 사용하지 마.

[출력 형식 예시]
```json
{{
    "recommend": "대화의 맥락에 따라 이런 챌린지를 추천합니다.",
    "challenges": [
        {{"title": "첫번째 챌린지:", "description": "간단한 설명"}},
        {{"title": "두번째 챌린지:", "description": "간단한 설명"}},
        {{"title": "세번째 챌린지:", "description": "간단한 설명"}}
    ]
}}
```
"""
)

def get_llm_response(query: str, chat_history: List[Dict[str, str]], context: str, category: str) -> Generator[Dict[str, Any], None, None]:
    """
    주어진 컨텍스트와 대화 기록을 바탕으로 vLLM에 스트리밍 요청을 보내고,
    모델의 자연스러운 띄어쓰기가 포함된 응답을 반환합니다.
    """
    memory = ConversationBufferMemory(return_messages=True)
    for msg in chat_history:
        if msg['role'] == 'user':
            memory.chat_memory.add_user_message(msg['content'])
        else:
            memory.chat_memory.add_ai_message(msg['content'])
    messages = memory.chat_memory.messages

    # custom_prompt 변수명을 사용하도록 수정
    final_prompt = custom_prompt.format(
        query=query,
        messages=messages,
        context=context,
        category=category
    )

    logger.info(f"[vLLM 호출] 프롬프트 길이: {len(final_prompt)}")
    url = "http://localhost:8800/v1/chat/completions"
    payload = {
        "model": "/home/ubuntu/mistral_finetuned_v5/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/0d4b76e1efeb5eb6f6b5e757c79870472e04bd3a",
        "messages": [{"role": "user", "content": final_prompt}],
        "stream": True, "max_tokens": 1024, "temperature": 0.7, "do_sample": True
    }

    response_completed = False
    full_response = ""
    streaming_buffer = ""
    recommend_sentence_finished = False
    full_cleaned_text_stream = ""
    
    try:
        with httpx.stream("POST", url, json=payload, timeout=60.0) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data: "):
                    json_str = line[len("data: "):]
                    if json_str.strip() == "[DONE]": break
                    
                    try:
                        json_data = json.loads(json_str)
                        token = json_data["choices"][0]["delta"].get("content", "")
                        if not token: continue
                        
                        logger.info(f"토큰 수신: {token[:20]}...")
                        full_response += token
                        streaming_buffer += token
                        
                        if ' ' in streaming_buffer:
                            parts = streaming_buffer.rsplit(' ', 1)
                            to_flush = parts[0] + ' '
                            streaming_buffer = parts[1]

                            if to_flush:
                                cleaned_text = re.sub(r'"(recommend|challenges|title|description)":\s*("|\')?', '', to_flush)
                                cleaned_text = cleaned_text.replace("```json", "").replace("```", "")
                                cleaned_text = re.sub(r'["\']', '', cleaned_text)
                                cleaned_text = re.sub(r'[\[\]{}$]', '', cleaned_text)
                                cleaned_text = re.sub(r',\s*$', '', cleaned_text)
                                
                                if cleaned_text.strip() and not response_completed:
                                    challenge_start_match = re.search(r'(2\.|3\.)', cleaned_text)
                                    if challenge_start_match:
                                        start_index = challenge_start_match.start()
                                        part_before_challenge = cleaned_text[:start_index]
                                        part_after_challenge = cleaned_text[start_index:]

                                        if part_before_challenge.strip():
                                            yield {"event": "challenge", "data": json.dumps({"status": 200, "message": "토큰 생성", "data": part_before_challenge}, ensure_ascii=False)}
                                        yield {"event": "challenge", "data": json.dumps({"status": 200, "message": "토큰 생성", "data": "\n\n"}, ensure_ascii=False)}
                                        yield {"event": "challenge", "data": json.dumps({"status": 200, "message": "토큰 생성", "data": part_after_challenge}, ensure_ascii=False)}
                                        full_cleaned_text_stream += cleaned_text
                                    else:
                                        yield {"event": "challenge", "data": json.dumps({"status": 200, "message": "토큰 생성", "data": cleaned_text}, ensure_ascii=False)}
                                        full_cleaned_text_stream += cleaned_text
                                    
                                    recommend_endings = ["추천합니다.", "추천드려요.", "추천해요.", "권장합니다."]
                                    if not recommend_sentence_finished and any(full_cleaned_text_stream.strip().endswith(ending) for ending in recommend_endings):
                                        yield {"event": "challenge", "data": json.dumps({"status": 200, "message": "토큰 생성", "data": "\n\n"}, ensure_ascii=False)}
                                        recommend_sentence_finished = True
                                
                    except json.JSONDecodeError:
                        logger.warning(f"JSON 디코딩 실패: {json_str}")
                        continue

            if streaming_buffer:
                cleaned_text = re.sub(r'"(recommend|challenges|title|description)":\s*("|\')?', '', streaming_buffer)
                cleaned_text = cleaned_text.replace("```json", "").replace("```", "")
                cleaned_text = re.sub(r'["\']', '', cleaned_text)
                cleaned_text = re.sub(r'[\[\]{}$]', '', cleaned_text)
                cleaned_text = re.sub(r',\s*$', '', cleaned_text)
                
                if cleaned_text.strip() and not response_completed:
                    yield {"event": "challenge", "data": json.dumps({"status": 200, "message": "토큰 생성", "data": cleaned_text.strip()}, ensure_ascii=False)}
                    full_cleaned_text_stream += cleaned_text.strip()
                    recommend_endings = ["추천합니다.", "추천드려요.", "추천해요.", "권장합니다."]
                    if not recommend_sentence_finished and any(full_cleaned_text_stream.strip().endswith(ending) for ending in recommend_endings):
                        yield {"event": "challenge", "data": json.dumps({"status": 200, "message": "토큰 생성", "data": "\n\n"}, ensure_ascii=False)}
                        recommend_sentence_finished = True

        logger.info("스트리밍 완료. 전체 응답 파싱 시작.")
        
        match = re.search(r'```json\s*([\s\S]*?)\s*```', full_response)
        if match:
            json_str = match.group(1)
        else:
            start_idx = full_response.find('{')
            end_idx = full_response.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = full_response[start_idx:end_idx]
            else:
                raise ValueError("응답에서 유효한 JSON 객체를 찾을 수 없습니다.")
        
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)

        try:
            parsed_data = json.loads(json_str)
            
            if not response_completed:
                response_completed = True
                yield {
                    "event": "close",
                    "data": json.dumps({
                        "status": 200,
                        "message": "모든 챌린지 추천 완료",
                        "data": parsed_data
                    }, ensure_ascii=False)
                }
        except json.JSONDecodeError as e:
            logger.error(f"최종 JSON 파싱 실패: {e}")
            logger.error(f"파싱 시도한 문자열: {json_str}")
            if not response_completed:
                response_completed = True
                yield {"event": "error", "data": json.dumps({"status": 500, "message": f"JSON 파싱 실패: {e}", "data": full_response}, ensure_ascii=False)}

    except httpx.HTTPStatusError as e:
        logger.error(f"vLLM 서버 오류: {e.response.status_code} - {e.response.text}")
        if not response_completed:
            yield {"event": "error", "data": json.dumps({"status": 500, "message": f"vLLM 서버 오류: {e.response.text}"})}
    except Exception as e:
        logger.error(f"[vLLM 호출 실패] {str(e)}")
        if not response_completed:
            yield {"event": "error", "data": json.dumps({"status": 500, "message": f"vLLM 호출 실패: {str(e)}"}, ensure_ascii=False)}

def process_chat(sessionId: str, query: str, base_info_category: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
    """대화 상태를 관리하고 스트리밍 응답을 생성하는 메인 함수"""
    logger.info(f"🚀🚀🚀 자유 대화 스트리밍 시작 🚀�🚀")
    logger.info(f"세션 ID: {sessionId}, 사용자 질문: {query}")

    # 1. 세션 상태 가져오기 또는 초기화
    if sessionId not in conversation_states:
        initial_category = base_info_category or "제로웨이스트"
        logger.info(f"새로운 세션 감지. 카테고리 초기화: {initial_category}")
        conversation_states[sessionId] = {
            "messages": [], "category": initial_category, "base_category": initial_category,
            "current_query": "", "context": "", "response": "", "should_continue": True,
            "error": None, "docs": None, "sessionId": sessionId
        }
    
    state = conversation_states[sessionId]
    state['messages'].append({'role': 'user', 'content': query})
    state['current_query'] = query

    # 2. 카테고리, 위치, 직업 변경 로직
    category_changed = False
    query_lower = query.lower()

    # 🔍 위치(location) 및 직업(workType) 자동 추출 로직 추가
    # 위치 관련 키워드 매핑
    location_keywords = {
        "도시": ["도시", "시내", "건물", "아파트"],
        "해안가": ["바다", "해변", "해안", "연안", "해안가"],
        "산": ["산", "산간", "등산", "고지대"],
        "농촌": ["농촌", "들판", "논", "밭", "시골"]
    }

    # 직업(근무 형태) 관련 키워드 매핑
    work_type_keywords = {
        "사무직": ["사무실", "오피스", "컴퓨터", "앉아서", "회의", "문서"],
        "영업직": ["영업", "판매", "고객", "외근", "출장", "상담"],
        "현장직": ["현장", "작업", "건설", "노동", "현장에서"],
        "재택근무": ["재택", "집", "원격", "재택근무", "홈오피스"]
    }

    # 위치 자동 변경 (자유 채팅 내용에 포함된 키워드 기반)
    for loc, keywords in location_keywords.items():
        if any(kw in query_lower for kw in keywords):
            state["location"] = loc
            logger.info(f"💡 위치 변경 감지: {loc}")
            break

    # 직업 자동 변경 (자유 채팅 내용에 포함된 키워드 기반)
    for work, keywords in work_type_keywords.items():
        if any(kw in query_lower for kw in keywords):
            state["workType"] = work
            logger.info(f"💡 직업 변경 감지: {work}")
            break

    # 기존 카테고리 변경 로직
    if any(keyword in query_lower for keyword in ["원래", "처음", "이전", "원래대로", "기존"]):
        state["category"] = state["base_category"]
        category_changed = True
        logger.info(f"카테고리 원복: {state['category']}")
    elif any(keyword in query_lower for keyword in ["아무", "아무거나", "다른거", "새로운거", "딴거", "다른"]):
        available_categories = [cat for cat in label_mapping.keys() if cat != state["category"]]
        if not available_categories:
            available_categories = list(label_mapping.keys())
        state["category"] = random.choice(available_categories)
        category_changed = True
        logger.info(f"카테고리 랜덤 변경: {state['category']}")
    else:
        for cat, keywords in category_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                state["category"] = cat
                category_changed = True
                logger.info(f"키워드 기반 카테고리 변경: {state['category']}")
                break

    if not category_changed:
        logger.info(f"카테고리 변경 없음. 기존 카테고리 유지: {state['category']}")

    # 3. RAG 문서 검색
    try:
        docs = retriever.get_relevant_documents(query)
        state["docs"] = docs
        state["context"] = "\n".join([doc.page_content for doc in docs])
        if not docs:
            logger.warning("관련된 챌린지 정보를 찾을 수 없습니다.")
    except Exception as e:
        logger.error(f"컨텍스트 검색 중 오류 발생: {e}")
        yield {"event": "error", "data": json.dumps({"status": 500, "message": f"컨텍스트 검색 중 오류 발생: {e}"}, ensure_ascii=False)}
        return

    # 4. LLM 스트리밍 호출 및 결과 처리
    final_response_data = None
    full_ai_response = ""
    
    for event in get_llm_response(state['current_query'], state['messages'], state['context'], state['category']):
        event_data_str = event.get("data", "{}")
        event_data = json.loads(event_data_str)
        
        if event['event'] == 'challenge':
            full_ai_response += event_data.get("data", "")
        elif event['event'] == 'close':
            final_response_data = event_data.get("data")

        yield event

    # 5. 스트리밍 종료 후 대화 기록 업데이트
    if final_response_data:
        eng_label = label_mapping.get(state['category'], "etc")
        if isinstance(final_response_data, dict) and "challenges" in final_response_data:
            for challenge in final_response_data["challenges"]:
                challenge["category"] = eng_label
        # 응답 최상위에 현재 카테고리 정보를 항상 포함시킴
        if isinstance(final_response_data, dict):
            final_response_data["category"] = state["category"]
        final_response_str = json.dumps(final_response_data, ensure_ascii=False)
        state['messages'].append({'role': 'assistant', 'content': final_response_str})
        state['response'] = final_response_str
    elif full_ai_response:
        state['messages'].append({'role': 'assistant', 'content': full_ai_response})
        state['response'] = full_ai_response
    
    logger.info(f"🚀🚀🚀 자유 대화 스트리밍 종료 🚀🚀🚀")

def clear_conversation(sessionId: str):
    """대화 기록 삭제"""
    if sessionId in conversation_states:
        del conversation_states[sessionId]
        logger.info(f"세션 ID {sessionId}의 대화 기록이 삭제되었습니다.")

def get_conversation_history(sessionId: str) -> List[Dict[str, str]]:
    """대화 기록 조회"""
    return conversation_states.get(sessionId, {}).get("messages", [])