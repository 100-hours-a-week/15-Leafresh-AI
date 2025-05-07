# main.py
from dotenv import load_dotenv
import uvicorn
from chatbot_router import router as chatbot_router
from fastapi import FastAPI

load_dotenv()

app = FastAPI()
app.include_router(chatbot_router)
uvicorn.run(app, host="0.0.0.0", port=8000)