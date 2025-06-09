from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_qdrant import Qdrant
from langchain_community.embeddings import SentenceTransformerEmbeddings
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence, Optional, Dict, List, Generator, Any
from Text.LLM.model.chatbot.chatbot_constants import label_mapping, ENV_KEYWORDS, BAD_WORDS
from huggingface_hub import InferenceClient
from langchain_core.language_models import BaseLLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import LLMResult
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
import os
import json
import random
import re
import requests
from requests.exceptions import HTTPError
from transformers import TextIteratorStreamer
import threading

load_dotenv()

class HuggingFaceInferenceLLM(BaseLLM):
    client: InferenceClient
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        # 대화 형식으로 변환
        messages = [{"role": "user", "content": prompt}]
        response = self.client.text_generation(
            prompt,
            max_new_tokens=1024,
            temperature=0.7,
            stop=stop or ["</s>", "[INST]"],
            do_sample=True,
            model=MODEL_NAME
        )
        return response

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> LLMResult:
        generations = []
        for prompt in prompts:
            response = self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
            generations.append([{"text": response}])
        return LLMResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        return "huggingface_inference"

# 환경 변수에서 값 가져오기
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEYMAC")
MODEL_NAME = os.getenv("HF_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")

# Hugging Face Inference Client 초기화
client = InferenceClient(token=HF_API_KEY)
llm = HuggingFaceInferenceLLM(client=client)

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

vectorstore = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embedding=embedding_model
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
    template=f"""<s>[INST] 당신은 환경 보호 챌린지를 추천하는 AI 어시스턴트입니다.
다음 문서와 이전 대화 기록을 참고하여 사용자에게 적절한 친환경 챌린지를 3개 추천해주세요.

이전 대화 기록:
{{messages}}

문서:
{{context}}

현재 요청:
{{query}}

JSON 포맷:
{escaped_format}

응답은 반드시 한글로 위 JSON형식 그대로 출력하세요. [/INST]</s>
"""
)

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
    sessionId: str         # 세션 ID
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

def format_sse_response(event: str, data: Dict[str, Any]) -> str:
    """SSE 응답 형식으로 변환"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

def get_llm_response(prompt: str) -> Generator[Dict[str, Any], None, None]:
    """LLM 응답을 SSE 형식으로 반환 (챌린지 단위 스트리밍)"""
    if not HF_API_KEY:
        yield {
            "event": "error",
            "data": json.dumps({
                "status": 401,
                "message": "Hugging Face API 키가 설정되지 않았습니다. .env 파일에 HUGGINGFACE_API_KEYMAC를 설정해주세요.",
                "data": None
            }, ensure_ascii=False)
        }
        return

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json; charset=utf-8"
    }
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 1024,
            "temperature": 0.7,
            "do_sample": True,
            "stop": ["</s>", "[INST]"],
            "return_full_text": False,
            "top_p": 0.95,
            "repetition_penalty": 1.1
        },
        "stream": True
    }

    try:
        print(f"Sending request to API with prompt: {prompt[:200]}...")
        response = requests.post(API_URL, headers=headers, json=payload, stream=True)
        response.raise_for_status()
        print(f"Response status code: {response.status_code}")
        
    except HTTPError as http_err:
        status_code = http_err.response.status_code
        try:
            error_detail = http_err.response.json()
            message = error_detail.get("error", str(http_err))
            if status_code == 401:
                message = "Hugging Face API 키가 유효하지 않습니다. API 키를 확인해주세요."
            elif status_code == 503:
                message = "모델이 현재 로딩 중입니다. 잠시 후 다시 시도해주세요."
            elif status_code == 422:
                message = "API 요청 형식이 잘못되었습니다. 프롬프트 형식을 확인해주세요."
        except json.JSONDecodeError:
            message = str(http_err)

        yield {
            "event": "error",
            "data": json.dumps({
                "status": status_code,
                "message": f"API 호출 오류: {message}",
                "data": None
            }, ensure_ascii=False)
        }
        return
    except Exception as e:
        yield {
            "event": "error",
            "data": json.dumps({
                "status": 500,
                "message": f"LLM 호출 중 예상치 못한 오류 발생: {str(e)}",
                "data": None
            }, ensure_ascii=False)
        }
        return

    try:
        full_response = ""
        json_buffer = ""
        in_json = False
        
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
                
            try:
                # SSE 형식에서 data: 접두사 제거
                if line.startswith("data: "):
                    line = line[6:]  # "data: " 제거
                
                data = json.loads(line)
                
                # 토큰 처리
                if isinstance(data, dict) and "token" in data:
                    token_data = data["token"]
                    if isinstance(token_data, dict) and "text" in token_data:
                        token_text = token_data["text"]
                    else:
                        continue
                        
                    if not token_text:
                        continue
                        
                    # 한글 인코딩 처리
                    try:
                        # UTF-8로 디코딩 시도
                        token_text = token_text.encode('latin1').decode('utf-8')
                    except:
                        try:
                            # 다른 인코딩 시도
                            token_text = token_text.encode('latin1').decode('cp949')
                        except:
                            continue
                        
                    full_response += token_text
                    
                    # 토큰 단위로 스트리밍
                    yield {
                        "event": "token",
                        "data": json.dumps({
                            "token": token_text
                        }, ensure_ascii=False)
                    }
                    
                    # JSON 내용 시작 감지
                    if "```json" in token_text:
                        in_json = True
                        json_buffer = ""
                        continue
                        
                    # JSON 내용이 아닌 경우 무시
                    if not in_json:
                        continue
                        
                    # JSON 내용 처리
                    if token_text.strip():
                        json_buffer += token_text
                        
                        # JSON 객체가 완성되었는지 확인
                        if json_buffer.strip().endswith("```"):
                            # 마지막 ``` 제거
                            json_buffer = json_buffer.strip()[:-3]
                            
                            # JSON 파싱 시도
                            try:
                                # 한글 인코딩 처리
                                try:
                                    json_buffer = json_buffer.encode('latin1').decode('utf-8')
                                except:
                                    json_buffer = json_buffer.encode('latin1').decode('cp949')
                                    
                                parsed_json = json.loads(json_buffer)
                                
                                if "challenges" in parsed_json:
                                    for idx, challenge in enumerate(parsed_json["challenges"]):
                                        challenge_data = {
                                            "index": idx + 1,
                                            "title": challenge.get("title", ""),
                                            "description": challenge.get("description", "")
                                        }
                                        yield {
                                            "event": "challenge",
                                            "data": json.dumps(challenge_data, ensure_ascii=False)
                                        }
                            except Exception as e:
                                print(f"JSON 파싱 실패: {str(e)}")
                                print(f"문제의 JSON: {json_buffer}")
                            
                            # JSON 처리 완료
                            in_json = False
                            json_buffer = ""
                
                # 전체 응답 처리
                elif isinstance(data, dict) and "generated_text" in data and data["generated_text"]:
                    # 전체 응답 파싱 시도
                    try:
                        # 한글 인코딩 처리
                        try:
                            full_response = full_response.encode('latin1').decode('utf-8')
                        except:
                            full_response = full_response.encode('latin1').decode('cp949')
                            
                        parsed_data = rag_parser.parse(full_response.strip())
                        
                        # 최종 응답이 완료되면 파싱된 데이터와 함께 완료 이벤트 전송
                        yield {
                            "event": "complete",
                            "data": json.dumps({
                                "status": 200,
                                "message": "모든 응답 완료",
                                "data": parsed_data
                            }, ensure_ascii=False)
                        }
                    except Exception as e:
                        print(f"파싱 실패: {str(e)}")
                        print(f"전체 응답: {full_response}")
                        yield {
                            "event": "error",
                            "data": json.dumps({
                                "status": 500,
                                "message": f"파싱 실패: {str(e)}",
                                "data": None
                            }, ensure_ascii=False)
                        }
                        return

            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON: {line}")
                continue

    except Exception as e:
        print(f"ERROR: get_llm_response 함수에서 예외 발생: {e}")
        yield {
            "event": "error",
            "data": json.dumps({
                "status": 500,
                "message": f"응답 처리 중 예상치 못한 오류 발생: {str(e)}",
                "data": None
            }, ensure_ascii=False)
        }
        return

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
        
        # Hugging Face API로 응답 생성
        prompt = custom_prompt.format(
            context=state["context"],
            query=state["current_query"],
            messages=messages,
            category=category
        )
        
        response = ""
        for data_payload in get_llm_response(prompt):
            event_type = data_payload.get("event_type", "message")
            if event_type == "token":
                response += data_payload.get("data", {}).get("token", "")
            elif event_type == "error":
                state["error"] = data_payload.get("message", "오류 발생")
                state["should_continue"] = False
                break
        
        if state["should_continue"]:
            # 필수 필드 검증
            if "recommend" not in json.loads(response) or "challenges" not in json.loads(response):
                raise ValueError("응답에 필수 필드가 없습니다.")
            
            # challenges가 문자열인 경우 배열로 변환
            if isinstance(json.loads(response).get("challenges"), str):
                challenges = parse_challenges_string(json.loads(response)["challenges"])
                json.loads(response)["challenges"] = challenges
            
            # challenges가 리스트가 아닌 경우 처리
            if not isinstance(json.loads(response).get("challenges"), list):
                raise ValueError("challenges는 리스트 형태여야 합니다.")
            
            # 현재 카테고리 정보로 챌린지 데이터 업데이트
            for challenge in json.loads(response)["challenges"]:
                challenge["category"] = eng_label
                challenge["label"] = kor_label
            
            state["response"] = json.dumps(json.loads(response), ensure_ascii=False)
            print(f"Final response with category: {category}, eng: {eng_label}, kor: {kor_label}")
            
        else:
            print(f"응답 검증 오류: {state['error']}")
            state["response"] = json.dumps({
                "recommend": "죄송합니다. 요청을 처리하는 중에 오류가 발생했습니다.",
                "challenges": []
            }, ensure_ascii=False)
        
        return state
        
    except Exception as e:
        print(f"응답 생성 중 오류 발생: {str(e)}")
        state["error"] = f"응답 생성 중 오류 발생: {str(e)}"
        state["should_continue"] = False
        return state

def handle_error(state: ChatState) -> ChatState:
    """오류 처리"""
    if state["error"]:
        state["response"] = json.dumps({
            "recommend": "죄송합니다. 요청을 처리하는 중에 오류가 발생했습니다.",
            "challenges": []
        }, ensure_ascii=False)
    return state

def create_chat_graph():
    """대화 그래프 생성"""
    workflow = StateGraph(ChatState)
    
    # 노드 추가
    workflow.add_node("validate_query", validate_query)
    workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("generate_response", generate_response)
    workflow.add_node("handle_error", handle_error)
    
    # 엣지 추가
    workflow.add_edge("validate_query", "retrieve_context")
    workflow.add_edge("retrieve_context", "generate_response")
    workflow.add_edge("generate_response", END)
    workflow.add_edge("handle_error", END)
    
    # 조건부 엣지 추가
    workflow.add_conditional_edges(
        "validate_query",
        lambda x: "retrieve_context" if x["should_continue"] else "handle_error"
    )
    
    workflow.add_conditional_edges(
        "retrieve_context",
        lambda x: "generate_response" if x["should_continue"] else "handle_error"
    )
    
    workflow.add_conditional_edges(
        "generate_response",
        lambda x: END if x["should_continue"] else "handle_error"
    )
    
    # 시작 노드 설정
    workflow.set_entry_point("validate_query")
    
    return workflow.compile()

# 대화 그래프 인스턴스 생성
chat_graph = create_chat_graph()

# 대화 상태 저장소
conversation_states = {}

def process_chat(sessionId: str, query: str, base_info_category: Optional[str] = None) -> str:
    """대화 처리"""
    # 세션 상태 초기화 또는 가져오기
    if sessionId not in conversation_states:
        conversation_states[sessionId] = {
            "messages": [],
            "category": base_info_category,
            "base_category": base_info_category
        }
    
    # 현재 상태 가져오기
    current_state = conversation_states[sessionId]
    
    # 새로운 상태 생성
    state = {
        "messages": current_state["messages"],
        "current_query": query,
        "context": "",
        "response": "",
        "should_continue": True,
        "error": None,
        "docs": None,
        "sessionId": sessionId,
        "category": current_state["category"],
        "base_category": current_state["base_category"]
    }
    
    # 대화 그래프 실행
    result = chat_graph.invoke(state)
    
    # 응답이 성공적으로 생성된 경우
    if result["should_continue"] and result["response"]:
        # 대화 기록 업데이트
        current_state["messages"].append(f"User: {query}")
        current_state["messages"].append(f"Assistant: {result['response']}")
        
        # 대화 기록이 너무 길어지면 오래된 메시지 제거
        if len(current_state["messages"]) > 10:
            current_state["messages"] = current_state["messages"][-10:]
        
        return result["response"]
    
    # 오류 발생 시
    return result["response"]

def clear_conversation(sessionId: str):
    """대화 기록 초기화"""
    if sessionId in conversation_states:
        conversation_states[sessionId]["messages"] = []

def get_conversation_history(sessionId: str) -> List[str]:
    """대화 기록 조회"""
    if sessionId in conversation_states:
        return conversation_states[sessionId]["messages"]
    return []
