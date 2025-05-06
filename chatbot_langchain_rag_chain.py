from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_qdrant import Qdrant
from langchain_community.embeddings import SentenceTransformerEmbeddings
from qdrant_client import QdrantClient
from langchain_google_vertexai import VertexAI
from dotenv import load_dotenv
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import os

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

retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # 사용자 질문으로 부터 가장 유사한 3개 문서 검색

# RAG 방식 챌린지 추천을 위한 Output Parser 정의
rag_response_schemas = [
    ResponseSchema(name="recommend", description="추천 텍스트(예: '이런 챌린지를 추천합니다.')"),
    ResponseSchema(name="challenges", description="추천 챌린지 리스트, 각 항목은 title, description 포함")
]

rag_parser = StructuredOutputParser.from_response_schemas(rag_response_schemas) # LangChain의 StructuredOutputParser를 사용하여 JSON 포맷을 정의
escaped_format = rag_parser.get_format_instructions().replace("{", "{{").replace("}", "}}")# JSON 포맷을 이스케이프 처리

# RAG 방식 챌린지 추천을 위한 PromptTemplate 정의
custom_prompt = PromptTemplate(
    input_variables=["context", "query"],
    template=f"""
다음 문서를 반드시 참고하여 사용자에게 적절한 친환경 챌린지를 3개 추천해주세요.

문서:
{{context}}

요청:
{{query}}

JSON 포맷:
{escaped_format}
- challenges는 반드시 리스트([]) 형태로 출력하세요.
- 문자열 형태로 묶지 말고, 리스트 그대로 출력하세요.
"""
)

# LLM 초기화 (VertexAI)
llm = VertexAI(model_name="gemini-2.0-flash", temperature=0.7)
# LLMChain 체인 생성 (retriever는 app_router에서 별도 사용)
qa_chain = LLMChain(
    llm=llm,
    prompt=custom_prompt
)