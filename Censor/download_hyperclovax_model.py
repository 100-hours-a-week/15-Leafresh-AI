from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from dotenv import load_dotenv

load_dotenv()

model_id = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
local_dir = "./hyperclovax_model"
hf_token = os.getenv("HF_TOKEN")  

print("--- 모델 다운로드 및 저장 시작 ---")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    cache_dir=local_dir,
    torch_dtype=torch.float16,
    device_map="auto",
    token=hf_token
)

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    cache_dir=local_dir,
    token=hf_token
)

print("--- 모델 & 토크나이저 저장 완료 ---")