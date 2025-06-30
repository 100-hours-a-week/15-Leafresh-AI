from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch, io

router = APIRouter()

# 로컬에 저장된 모델 경로
MODEL_PATH = "./llava_model"
processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

@router.post("/verify-llava")
async def verify_llava_image(
    image: UploadFile = File(...),
    prompt: str = Form(...)
):
    try:
        # 이미지 디코딩
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 프롬프트 및 이미지 입력 처리
        inputs = processor(prompt, images=pil_image, return_tensors="pt").to("cuda", torch.float16)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=100)

        output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        return JSONResponse(content={"result": output})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

