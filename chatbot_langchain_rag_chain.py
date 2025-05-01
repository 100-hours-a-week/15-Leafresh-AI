# chatbot_langchain_rag_chain.py
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import SentenceTransformerEmbeddings
from qdrant_client import QdrantClient
from langchain_google_vertexai import VertexAI
from dotenv import load_dotenv
import os

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "challenge-collection"

# LLM 초기화 (VertexAI)
llm = VertexAI(model_name="gemini-1.5-flash", temperature=0.4)

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Qdrant(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embeddings=embedding_model
)

retriever = vectorstore.as_retriever()

# PromptTemplate
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
다음 문서를 참고하여 사용자 요청에 맞는 친환경 챌린지를 JSON 형식으로 3개 추천해주세요.

문서:
{context}

요청:
{question}

아래 포맷을 반드시 지켜 JSON만 출력해주세요.  
추가 설명 없이 순수 JSON만 반환해야 합니다.
절대로 ```json 또는 ``` 와 같은 마크다운 코드블록을 사용하지 마세요.

{{
  "status": 200,
  "message": "성공!",
  "data": {{
    "recommand": "설명 텍스트",
    "challenges": [
      {{"title": "챌린지 이름", "description": "설명"}},
      {{"title": "챌린지 이름", "description": "설명"}},
      {{"title": "챌린지 이름", "description": "설명"}}
    ]
  }}
}}
"""
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt}
)