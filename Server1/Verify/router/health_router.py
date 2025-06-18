from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

@router.get("/health")
async def health_check():
    try:
        return JSONResponse(
            status_code=200,
            content={
                "status": 200,
                "message": "OK",
                "data": ""
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": 500,
                "message": f"error: {e}",
                "data": ""
            }
        )
