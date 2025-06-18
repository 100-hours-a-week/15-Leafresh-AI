from dotenv import load_dotenv
from fastapi import FastAPI

from fastapi.exceptions import RequestValidationError, HTTPException
from router.censorship_router import router as censorship_router
from router.censorship_router import validation_exception_handler, http_exception_handler

from router.health_router import router as health_router

load_dotenv()

app = FastAPI()

app.include_router(censorship_router)
app.include_router(health_router)

# censorship model exceptions (422, 500, 503 etc.)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(HTTPException, http_exception_handler)