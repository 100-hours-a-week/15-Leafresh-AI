# chatbot_langchain_rag_chain.py
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import SentenceTransformerEmbeddings
from qdrant_client import QdrantClient
from langchain_gemini_llm import GeminiLLM
from dotenv import load_dotenv
import os

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "challenge-collection"

llm = GeminiLLM(
    credentials_path="./kakao-project-457106-b926aa186fc4.json",
    project="kakao-project-457106",
    location="us-central1",
    model_name="gemini-1.5-pro"
)

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Qdrant(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

if __name__ == "__main__":
    question = "플라스틱 줄이는 방법 뭐 있어?"
    response = qa_chain.run(question)
    print(f"🤖 응답: {response}")