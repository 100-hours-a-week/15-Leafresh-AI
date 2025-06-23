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
from transformers import TextIteratorStreamer, LogitsProcessorList, InfNanRemoveLogitsProcessor
import torch
import os
import json
import random
import re
import threading
import logging
from fastapi import HTTPException
import gc
from Text.LLM.model.chatbot.shared_model import shared_model

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

# 공유 모델 사용
model = shared_model.model
tokenizer = shared_model.tokenizer

logger.info("Using shared Mistral model for free-text chatbot")

# Qdrant 설정
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

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# RAG 방식 챌린지 추천을 위한 Output Parser 정의
rag_response_schemas = [
    ResponseSchema(name="recommend", description="추천 텍스트를 한글로 한 문장으로 출력해주고 실제 줄바꿈(엔터)으로 구분해 주세요.(예: '이런 챌린지를 추천합니다.')"),
    ResponseSchema(name="challenges", description="추천 챌린지 리스트, 각 항목은 title, description 포함, description은 한글로 한 문장으로 요약해주세요.")
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

주의사항:
1. 모든 속성 이름과 문자열 값은 반드시 큰따옴표(")로 둘러싸야 합니다.
2. recommend 필드에는 {{category}} 관련 추천 문구를 포함해야 합니다.
3. 각 title 내용은 번호를 붙이고, 실제 줄바꿈(엔터)으로 구분해 주세요.

출력 형식 예시:
{escaped_format}

반드시 위 JSON 형식 그대로 반드시 한글로 한번만 출력하세요. [/INST]</s>
"""
)

def get_llm_response(prompt: str, category: str) -> Generator[Dict[str, Any], None, None]:
    """LLM 응답을 SSE 형식으로 반환 (서버에서 전체 파싱 후 전달)"""
    logger.info("==== [get_llm_response] 함수 진입 ====")
    logger.info(f"[get_llm_response]생성 시작 - 프롬프트 길이: {len(prompt)}")
    try:
        # 메모리 정리
        shared_model.cleanup_memory()
        logger.info("get_llm_response 시작 전 메모리 정리 완료")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        logger.info(f"토크나이저 입력 준비 완료. 입력 토큰 수: {inputs.input_ids.shape[1]}")
        # 스트리머 설정
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=None,
            decode_kwargs={
                "skip_special_tokens": True,
                "clean_up_tokenization_spaces": True,
                "errors": "ignore" # 디코딩 할 수 없는 바이트는 무시
            }
        )
        logits_processor = LogitsProcessorList([
            InfNanRemoveLogitsProcessor()
        ])

       # 모델 생성 설정 
        generation_kwargs = dict(
            inputs,  # 입력 텐서 (input_ids, attention_mask 등)
            streamer=streamer,  # 스트리밍 응답을 위한 TextIteratorStreamer 객체
            max_new_tokens=512,  # 토큰 수를 줄여서 안정성 향상
            temperature=0.3,  # 더 낮은 temperature로 일관성 향상
            do_sample=True,  # 확률적 샘플링 활성화
            top_p=0.9,  # nucleus sampling 추가
            top_k=50,  # top-k sampling 추가
            repetition_penalty=1.1,  # 반복 방지
            pad_token_id=tokenizer.eos_token_id, # 패딩 토큰 ID 설정 (Mistral은 EOS 토큰을 패딩으로 사용)
            logits_processor=logits_processor,
            eos_token_id=tokenizer.eos_token_id,  # EOS 토큰 명시적 설정
            early_stopping=True  # 조기 중단 활성화
        )
        logger.info("스레드 시작 및 모델 생성 시작.")

        if inputs.input_ids.shape[1] > 2048:
            raise ValueError(f"입력이 너무 깁니다. 최대 2048 토큰까지 허용됩니다. 현재: {inputs.input_ids.shape[1]} 토큰")

        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

         # 전체 응답 누적용
        full_response = ""
        logger.info("스트리밍 응답 대기 중...")
        response_completed = False  # 응답 완료 여부를 추적하는 플래그

        try:
            for new_text in streamer:
                if new_text and not response_completed:  # 응답이 완료되지 않은 경우에만 처리
                    full_response += new_text
                    logger.info(f"토큰 수신: {new_text[:20]}...")
                    
                    # 토큰 정제 - 순수 텍스트만 추출
                    cleaned_text = new_text
                    # JSON 관련 문자열 제거
                    cleaned_text = re.sub(r'"(recommend|challenges|title|description)":\s*("|\')?', '', cleaned_text)
                    # 마크다운 및 JSON 구조 제거
                    cleaned_text = cleaned_text.replace("```json", "").replace("```", "").strip()
                    cleaned_text = cleaned_text.replace("\\n", "\n")  # 문자열 → 줄바꿈
                    cleaned_text = re.sub(r'["\']', '', cleaned_text)  # 따옴표 제거
                    cleaned_text = re.sub(r'[\[\]{}]', '', cleaned_text)  # 괄호 제거
                    cleaned_text = re.sub(r',\s*$', '', cleaned_text)  # 끝의 쉼표 제거
                    # 불필요한 공백 제거
                    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
                    cleaned_text = cleaned_text.strip()
                    
                    if cleaned_text:
                        # SSE 응답 전송 - 순수 텍스트만 전송
                        try:
                            yield {
                                "event": "challenge",
                                "data": json.dumps({
                                    "status": 200,
                                    "message": "토큰 생성",
                                    "data": new_text
                                }, ensure_ascii=False)
                            }
                        except Exception as e:
                            logger.error(f"이벤트 전송 중 에러 발생: {str(e)}")
                            response_completed = True
                            continue

            # 스레드 완료 대기
            thread.join()
            
            # 토큰 캐시 정리
            if hasattr(streamer, 'token_cache'):
                streamer.token_cache = []
            
            # 전체 응답 파싱
            logger.info("스트리밍 완료. 전체 응답 파싱 시작.")
            try:
                # 전체 응답 로깅
                logger.info(f"전체 응답: {full_response}")
                
                # JSON 문자열 추출
                json_match = re.search(r"```json\n([\s\S]*?)\n```", full_response.strip())
                if json_match:
                    json_str = json_match.group(1).strip()
                    # logger.info(f"JSON 추출: {json_str}")
                else:
                    # 마크다운 코드 블록이 없는 경우, 마지막 JSON 객체 찾기
                    json_match = re.search(r'(\{[\s\S]*\})', full_response)
                    if json_match:
                        json_str = json_match.group(1).strip()
                        logger.info(f"JSON 객체 추출: {json_str}")
                    else:
                        raise ValueError("JSON을 찾을 수 없습니다")

                if not json_str.strip():
                    raise ValueError("JSON 문자열이 비어있습니다")

                # JSON 파싱 전 문자열 정제

                # 1. 객체와 배열의 마지막 쉼표 제거: {...,} → {...}
                json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

                # 2. 연속된 쉼표 제거: "a",, "b" → "a", "b"
                json_str = re.sub(r',\s*,', ',', json_str)

                # 3. 키와 값의 작은따옴표만 큰따옴표로 바꾸기
                # 3.1. 'recommend': → "recommend":
                json_str = re.sub(r"'(\w+)'\s*:", r'"\1":', json_str)

                # 3.2. "recommend": 'value' → "recommend": "value"
                json_str = re.sub(r':\s*\'([^\']*)\'', r': "\1"', json_str)

                # 4. 공백 정리
                json_str = re.sub(r'\s+', ' ', json_str).strip()

                # JSON 파싱
                try:
                    parsed_data_temp = json.loads(json_str)
                    # logger.info(f"JSON 파싱 성공: {parsed_data_temp}")
                    parsed_data = rag_parser.parse(json.dumps(parsed_data_temp))
                    logger.info(f"파싱 성공: {parsed_data}")
                except json.JSONDecodeError as e:
                    logger.error(f"JSON 파싱 실패: {str(e)}")
                    logger.error(f"파싱 시도한 문자열: {json_str}")
                    response_completed = True  # 파싱 실패 시 플래그 설정
                    raise
                    
                # 카테고리 정보 추가
                eng_label = label_mapping[category]
                if isinstance(parsed_data, dict) and "challenges" in parsed_data:
                    for challenge in parsed_data["challenges"]:
                        challenge["category"] = eng_label
                    logger.info(f"카테고리 추가 완료: {eng_label}")
                    logger.info(f"카테고리 추가된 챌린지 데이터: {parsed_data['challenges']}")
                    if not response_completed:  # 아직 응답이 완료되지 않았다면
                        response_completed = True 
                        yield {
                            "event": "close",
                            "data": json.dumps({
                                "status": 200,
                                "message": "모든 챌린지 추천 완료",
                                "data": parsed_data
                            }, ensure_ascii=False)
                        }
                else:
                    response_completed = True  # challenges 필드가 없는 경우 플래그 설정
                    raise ValueError("파싱된 데이터에 'challenges' 필드가 없습니다.")
                    
            except json.JSONDecodeError as e:
                logger.error(f"JSON 파싱 실패: {str(e)}")
                response_completed = True
                yield {
                    "event": "error",
                    "data": json.dumps({
                        "status": 500,
                        "message": f"JSON 파싱 실패: {str(e)}",
                        "data": None
                    }, ensure_ascii=False)
                }
        except Exception as e:
            logger.error(f"스트리밍 중 에러 발생: {str(e)}")
            response_completed = True
            yield {
                "event": "error",
                "data": json.dumps({
                    "status": 500,
                    "message": f"LLM 응답 생성 중 예외 발생: {str(e)}",
                    "data": None
                }, ensure_ascii=False)
            }
    except Exception as e:
        logger.error(f"=== 예외 발생 ===")
        logger.error(f"예외 타입: {type(e).__name__}")
        logger.error(f"예외 메시지: {str(e)}")
        
        yield {
            "event": "error",
            "data": json.dumps({
                "status": 500,
                "message": f"LLM 응답 생성 실패: {str(e)}",
                "data": None
            }, ensure_ascii=False)
        }
    finally:
        # 메모리 정리
        shared_model.cleanup_memory()
        logger.info("메모리 정리 완료")

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

def format_sse_response(event: str, data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "event": event,
        "data": json.dumps(data, ensure_ascii=False)
    }

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
        eng_label = label_mapping[category]
        logger.info(f"Adding category info - eng: {eng_label}")
        
        # 프롬프트 생성
        prompt = custom_prompt.format(
            context=state["context"],
            query=state["current_query"],
            messages=messages,
            category=category
        )
        
        # LLM 응답 생성 (스트리밍 방식 유지)
        full_response = ""
        for data_payload in get_llm_response(prompt, category):
            if isinstance(data_payload, dict) and "data" in data_payload:
                full_response += str(data_payload["data"])
            yield data_payload

        if state["should_continue"]:
            print(f"Raw LLM response: {full_response}")
            
            # JSON 파싱 시도
            try:
                response_text = full_response
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1]
                if "```" in response_text:
                    response_text = response_text.split("```")[0]
                response_text = response_text.strip()
                
                # JSON 파싱
                parsed_response = json.loads(response_text)
                print(f"Successfully parsed JSON response. Length: {len(response_text)}")
                
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
                
                state["response"] = json.dumps(parsed_response, ensure_ascii=False)
                print(f"Final response with category: {category}, eng: {eng_label}")
                
            except ValueError as e:
                print(f"응답 검증 오류: {str(e)}")
                state["error"] = str(e)
                state["should_continue"] = False
                return state
        else:
            print(f"응답 검증 오류: {state['error']}")
            state["response"] = json.dumps({
                "recommend": "죄송합니다. 요청을 처리하는 중에 오류가 발생했습니다.",
                "challenges": []
            }, ensure_ascii=False)
        
        # 대화 기록 업데이트 (비교 코드 방식으로 개선)
        state["messages"] = list(state["messages"]) + [
            f"User: {state['current_query']}",
            f"Assistant: {state['response']}"
        ]
        
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
            
        eng_label = label_mapping[current_category]
        
        # 챌린지 데이터에 현재 카테고리 정보 업데이트
        if "challenges" in response_data:
            for challenge in response_data["challenges"]:
                challenge["category"] = eng_label
        
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
    finally:
        # 메모리 정리
        shared_model.cleanup_memory()
        logger.info("process_chat 메모리 정리 완료")

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
