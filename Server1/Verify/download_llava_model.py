from transformers import AutoModelForVision2Seq, AutoProcessor
import torch
import os
from dotenv import load_dotenv

load_dotenv()

model_id = "llava-hf/llava-1.5-13b-hf"
local_dir = "./llava_model"             
hf_token = os.getenv("HF_TOKEN")

print("--- 모델 다운로드 및 저장 시작 ---")

model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    cache_dir=local_dir,
    torch_dtype=torch.float16,
    device_map="auto",
    token=hf_token
)

processor = AutoProcessor.from_pretrained(
    model_id,
    cache_dir=local_dir,
    token=hf_token
)

print("-- 모델 & 프로세서 저장 완료 ---")