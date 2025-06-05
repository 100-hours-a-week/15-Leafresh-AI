from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_qdrant import Qdrant
from langchain_community.embeddings import SentenceTransformerEmbeddings
from qdrant_client import QdrantClient
from langchain_google_vertexai import VertexAI
from dotenv import load_dotenv
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence, Optional, Dict, List
from Text.LLM.model.chatbot.chatbot_constants import label_mapping, ENV_KEYWORDS, BAD_WORDS
import os
import json
import random
import re

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embedding_model = SentenceTransformerEmbeddings(model_name="BAAI/bge-small-en-v1.5")

vectorstore = Qdrant(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embeddings=embedding_model
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # 사용자 질문으로 부터 가장 유사한 3개 문서 검색

# RAG 방식 챌린지 추천을 위한 Output Parser 정의
rag_response_schemas = [
    ResponseSchema(name="recommend", description="추천 텍스트를 한 문장으로 출력해줘.(예: '이런 챌린지를 추천합니다.')"),
    ResponseSchema(name="challenges", description="추천 챌린지 리스트, 각 항목은 title, description 포함, description은 한 문장으로 요약해주세요.")
]

# LangChain의 StructuredOutputParser를 사용하여 JSON 포맷을 정의
rag_parser = StructuredOutputParser.from_response_schemas(rag_response_schemas)

# JSON 포맷을 이스케이프 처리
escaped_format = rag_parser.get_format_instructions().replace("{", "{{").replace("}", "}}")

# RAG 방식 챌린지 추천을 위한 PromptTemplate 정의
custom_prompt = PromptTemplate(
    input_variables=["context", "query", "messages", "category"],
    template=f"""
다음 문서와 이전 대화 기록을 참고하여 사용자에게 적절한 친환경 챌린지를 3개 추천해주세요.

이전 대화 기록:
{{messages}}

문서:
{{context}}

현재 요청:
{{query}}

JSON 포맷:
{escaped_format}

응답은 반드시 위 JSON형식 그대로 출력하세요.


"""
)

# LLM 초기화 (VertexAI)
llm = VertexAI(model_name="gemini-2.0-flash", temperature=0.7)

# LLMChain 체인 생성 (retriever는 app_router에서 별도 사용)
qa_chain = LLMChain(
    llm=llm,
    prompt=custom_prompt
)

##########################################

# 대화 상태를 관리하기 위한 타입 정의
class ChatState(TypedDict):
    messages: Annotated[Sequence[str], "대화 기록"]
    current_query: str     # 사용자가 입력한 현재 질문
    context: str           # RAG 검색 자료
    response: str          # LLM 최종응답 
    should_continue: bool  # 대화 계속 여부
    error: Optional[str]   # 오류 메시지
    docs: Optional[list]   # 검색된 문서
    sessionId: str   # 세션 ID
    category: Optional[str]  # 현재 선택된 카테고리
    base_category: Optional[str]  # 원본 카테고리도 저장

def parse_challenges_string(challenges_str: str) -> list:
    """challenges 문자열을 파싱하여 리스트로 변환"""
    # 이미 리스트인 경우 그대로 반환
    if isinstance(challenges_str, list):
        return challenges_str
    
    # JSON 파싱 시도
    try:
        return json.loads(challenges_str)
    except:
        pass
    
    # 문자열이 아닌 경우 빈 리스트 반환
    if not isinstance(challenges_str, str):
        return []
    
    challenges = []
    current_challenge = {}
    
    # 줄 단위로 분리
    lines = challenges_str.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # 빈 줄 무시
        if not line:
            continue
            
        # 새로운 챌린지 시작
        if line.startswith('-') or line.startswith('*'):
            # 이전 챌린지가 있으면 추가
            if current_challenge and 'title' in current_challenge:
                challenges.append(current_challenge)
                current_challenge = {}
            
            # title 추출
            title_match = re.search(r'(?:title|제목)[\s:]*[\'"]?([^\'"]+)[\'"]?', line, re.IGNORECASE)
            if title_match:
                current_challenge['title'] = title_match.group(1).strip()
        
        # description 추출
        elif 'description' in line.lower() or '설명' in line:
            desc_match = re.search(r'(?:description|설명)[\s:]*[\'"]?([^\'"]+)[\'"]?', line, re.IGNORECASE)
            if desc_match:
                current_challenge['description'] = desc_match.group(1).strip()
    
    # 마지막 챌린지 추가
    if current_challenge and 'title' in current_challenge:
        challenges.append(current_challenge)
    
    return challenges

# 대화 그래프 노드 정의
def validate_query(state: ChatState) -> ChatState: # state는 챗봇의 현재 대화 상태를 담고 있는 딕셔너리
    """사용자 질문 유효성 검사"""
    if len(state["current_query"].strip()) < 5:
        state["error"] = "질문은 최소 5자 이상이어야 합니다."
        state["should_continue"] = False
    else:
        state["should_continue"] = True
    return state

def retrieve_context(state: ChatState) -> ChatState:
    """관련 컨텍스트 검색(RAG)"""
    if not state["should_continue"]:
        return state # 다음 단계로 진행할지를 결정하는 체크포인트 역할
    try:
        # RAG 검색 수행 (카테고리 필터 제거)
        docs = retriever.get_relevant_documents(state["current_query"])
        state["docs"] = docs
        state["context"] = "\n".join([doc.page_content for doc in docs])

        # 참조된 문서 로그 출력
        for idx, doc in enumerate(docs):
            print(f"[RAG 참조 문서 {idx+1}]")
            print(f"내용: {doc.page_content[:200]}")  # 너무 길면 일부만 출력
            print(f"메타데이터: {doc.metadata}")
        
        # 검색된 문서가 없는 경우
        if not docs:
            state["error"] = "관련된 챌린지 정보를 찾을 수 없습니다."
            state["should_continue"] = False
            
    except Exception as e:
        state["error"] = f"컨텍스트 검색 중 오류 발생: {str(e)}"
        state["should_continue"] = False
    
    return state


def generate_response(state: ChatState) -> ChatState:
    """응답 생성"""
    if not state["should_continue"]:
        return state
    try:
        messages = "\n".join(state["messages"])
        print(f"Generating response for query: {state['current_query']}")
        print(f"Current category in state: {state['category']}")
        
        category = state["category"]
        if category not in label_mapping:
            raise ValueError(f"잘못된 카테고리 값: {category}")
        eng_label, kor_label = label_mapping[category]
        
        response = qa_chain.invoke({
            "context": state["context"], # RAG 정보
            "query": state["current_query"],
            "messages": state["messages"],
            "category": category
        })
        
        print(f"Raw LLM response: {response['text']}")
        
        # JSON 파싱 시도
        try:
            response_text = response["text"]
            if "```json" in response_text:
                response_text = response_text.split("```json")[1]
            if "```" in response_text:
                response_text = response_text.split("```")[0]
            response_text = response_text.strip()
            
            # JSON 파싱 시도
            try:
                parsed_response = json.loads(response_text)
                print(f"Successfully parsed JSON response. Length: {len(response_text)}")
            except json.JSONDecodeError:
                raise ValueError("JSON 형식이 올바르지 않습니다.")
            
            # 필수 필드 검증
            if "recommend" not in parsed_response or "challenges" not in parsed_response:
                raise ValueError("응답에 필수 필드가 없습니다.")
            
            # challenges가 문자열인 경우 배열로 변환
            if isinstance(parsed_response.get("challenges"), str):
                challenges = parse_challenges_string(parsed_response["challenges"])
                parsed_response["challenges"] = challenges
            
            # challenges가 리스트가 아닌 경우 처리
            if not isinstance(parsed_response.get("challenges"), list):
                raise ValueError("challenges는 리스트 형태여야 합니다.")
            
            # 현재 카테고리 정보로 챌린지 데이터 업데이트
            for challenge in parsed_response["challenges"]:
                challenge["category"] = eng_label
                challenge["label"] = kor_label
            
            state["response"] = json.dumps(parsed_response, ensure_ascii=False)
            print(f"Final response with category: {category}, eng: {eng_label}, kor: {kor_label}")
            
        except ValueError as e:
            print(f"응답 검증 오류: {str(e)}")
            state["error"] = str(e)
            state["should_continue"] = False
            return state
        
        # 대화 기록 업데이트
        state["messages"] = list(state["messages"]) + [
            f"User: {state['current_query']}",
            f"Assistant: {state['response']}"
        ]
    except Exception as e:
        print(f"응답 생성 중 예외 발생: {str(e)}")
        state["error"] = f"응답 생성 중 오류 발생: {str(e)}"
        state["should_continue"] = False
    return state

def handle_error(state: ChatState) -> ChatState:
    """오류 처리"""
    if state["error"]:
        state["response"] = state["error"]
        # 오류 메시지도 대화 기록에 추가
        state["messages"] = list(state["messages"]) + [
            f"User: {state['current_query']}",
            f"Assistant: {state['error']}"
        ]
    return state

# 대화 그래프 구성
def create_chat_graph():
    workflow = StateGraph(ChatState)
    
    # 노드 추가
    workflow.add_node("validate", validate_query)
    workflow.add_node("retrieve", retrieve_context)
    workflow.add_node("generate", generate_response)
    workflow.add_node("handle_error", handle_error)
    
    # 엣지 연결
    workflow.add_edge("validate", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "handle_error")
    workflow.add_edge("handle_error", END)
    
    # 조건부 라우팅
    workflow.add_conditional_edges(
        "validate",
        lambda x: "retrieve" if x["should_continue"] else "handle_error"
    )
    
    # 시작 노드 설정
    workflow.set_entry_point("validate")
    
    return workflow.compile()

# 대화 그래프 인스턴스 생성
chat_graph = create_chat_graph()

# 대화 상태 저장소
conversation_states: Dict[str, ChatState] = {}

def process_chat(sessionId: str, query: str, base_info_category: Optional[str] = None) -> str:
    """대화 처리 함수"""
    print(f"\n=== Process Chat Start ===")
    print(f"Initial base_info_category: {base_info_category}")
    print(f"User query: {query}")
    print(f"Session ID: {sessionId}")

    # 이전 대화 상태 가져오기 또는 새로 생성
    if sessionId not in conversation_states:
        if not base_info_category:
            raise ValueError("새로운 세션은 base-info에서 카테고리가 필요합니다.")
        if base_info_category not in label_mapping:
            raise ValueError(f"잘못된 카테고리 값: {base_info_category}")
            
        print(f"New session detected. Initializing with category: {base_info_category}")
        conversation_states[sessionId] = {
            "messages": [],             # 대화 기록 
            "current_query": "",        # 사용자가 입력한 현재 질문
            "context": "",              # RAG 검색 자료
            "response": "",             # LLM 최종응답 
            "should_continue": True,    # 대화 진행 가능성 여부
            "error": None,
            "docs": None,               # 검색된 원본 문서 리스트 (Qdrant의 Document 객체들)
            "sessionId": sessionId,
            "category": base_info_category,  # base-info 카테고리 저장 -> 사용자에 요청에 따라 변경되는 카테고리
            "base_category": base_info_category  # 원본 카테고리도 저장
        }
        # 초기 카테고리 설정 로그
        conversation_states[sessionId]["messages"].append(f"Initial category set to {base_info_category}")
    else:
        print(f"현재 카테고리: {conversation_states[sessionId]['category']}")
    
    # 현재 상태 업데이트
    state = conversation_states[sessionId]
    state["current_query"] = query
    print(f"Current state category before random: {state['category']}")

    # 카테고리 변경 처리
    category_changed = False

    # 1. "원래 카테고리로" 요청 처리
    if any(keyword in query.lower() for keyword in ["원래", "처음", "이전", "원래대로","기존"]):
        if state["base_category"]:
            state["category"] = state["base_category"]
            state["messages"].append(f"Category restored to original: {state['base_category']}")
            category_changed = True

    # 2. "아무거나" 등의 요청 처리
    elif any(keyword in query.lower() for keyword in ["아무", "아무거나", "다른거", "새로운거", "딴거", "다른"]):
        available_categories = [cat for cat in label_mapping.keys() if cat != state["category"]]
        if not available_categories:
            available_categories = list(label_mapping.keys())
        
        sampled_category = random.choice(available_categories)
        state["category"] = sampled_category
        state["messages"].append(f"Category randomly selected: {sampled_category}")
        category_changed = True

    # 3. 특정 카테고리 요청 처리
    else:
        for category in label_mapping.keys():
            if category in query:
                state["category"] = category
                state["messages"].append(f"Category changed to {category}")
                category_changed = True
                break

    # 4. base-info 카테고리 처리
    if not category_changed and base_info_category and state["category"] != base_info_category:
        state["category"] = base_info_category
        state["messages"].append(f"Category changed to {base_info_category}")
        category_changed = True

    print(f"State category before chat_graph: {state['category']}")

    # 대화 그래프 실행
    result = chat_graph.invoke(state)
    
    # 응답 생성 시 현재 카테고리 정보 포함
    try:
        if not result["response"]:
            raise ValueError("응답이 비어있습니다.")
            
        response_data = json.loads(result["response"])
        current_category = result["category"]
        print(f"Current category in result: {current_category}")
        
        if current_category not in label_mapping:
            raise ValueError(f"잘못된 카테고리 값: {current_category}")
            
        eng_label, kor_label = label_mapping[current_category]
        
        # 챌린지 데이터에 현재 카테고리 정보 업데이트
        if "challenges" in response_data:
            for challenge in response_data["challenges"]:
                challenge["category"] = eng_label
                challenge["label"] = kor_label
        
        # 업데이트된 응답으로 result 수정
        result["response"] = json.dumps(response_data, ensure_ascii=False)
        
        # 상태 저장
        conversation_states[sessionId] = result
        print(f"Final state category: {result['category']}")
        
        return result["response"]
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 오류: {str(e)}")
        print(f"Raw response: {result.get('response', '')}")
        # 상태 저장
        conversation_states[sessionId] = result
        return json.dumps({
            "recommend": "죄송합니다. 응답을 처리하는 중에 오류가 발생했습니다.",
            "challenges": []
        }, ensure_ascii=False)
    except Exception as e:
        print(f"Error in response processing: {str(e)}")
        # 상태 저장
        conversation_states[sessionId] = result
        return json.dumps({
            "recommend": "죄송합니다. 응답을 처리하는 중에 오류가 발생했습니다.",
            "challenges": []
        }, ensure_ascii=False)

def clear_conversation(sessionId: str):
    """대화 기록 삭제"""
    if sessionId in conversation_states:
        del conversation_states[sessionId]

def get_conversation_history(sessionId: str) -> List[str]:
    """대화 기록 조회
    Args:
        sessionId: 사용자 세션 ID
    
    Returns:
        List[str]: 대화 기록 리스트
    """
    if sessionId in conversation_states:
        return conversation_states[sessionId]["messages"]
    return []
