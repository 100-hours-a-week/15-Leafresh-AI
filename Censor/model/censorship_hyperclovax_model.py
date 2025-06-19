import os, traceback, re
from dotenv import load_dotenv
from typing import List

from datetime import datetime
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# GCP model_dir = "/home/ubuntu/hyperclovax_model/models--naver-hyperclovax--HyperCLOVAX-SEED-Text-Instruct-1.5B/snapshots/543a1be9d6233069842ffce73aa56a232a4f457b"
# local model_dir = "./hyperclovax_model/models--naver-hyperclovax--HyperCLOVAX-SEED-Text-Instruct-1.5B/snapshots/543a1be9d6233069842ffce73aa56a232a4f457b"
class HyperClovaxModel :
    def __init__(self, model_dir = "/home/ubuntu/hyperclovax_model/models--naver-hyperclovax--HyperCLOVAX-SEED-Text-Instruct-1.5B/snapshots/543a1be9d6233069842ffce73aa56a232a4f457b"):
        load_dotenv()
        self.device = "cpu"

        print("[INFO] HyperCLOVA-X 모델 로딩 중 ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float32,                          # CPU 환경에서는 float16 사용 불가
        ).to(self.device)
        print("[INFO] 모델 로딩 완료")


    def generate_response(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=128,
                temperature=None,
                top_p=1.0,
                # top_k=32,
                eos_token_id=self.tokenizer.eos_token_id, # 종료 토큰 지정 < /s>
                pad_token_id=self.tokenizer.eos_token_id
            )

        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result.strip()
    

    def validate(self, challenge_name: str, start_date: str, end_date: str, existing: List[dict]):
        # 날짜 계산 함수 
        def dates_overlap(s1, e1, s2, e2):
            return max(s1, s2) <= min(e1, e2)
        
        # 공백 정규화 (여러 개의 공백을 하나로 줄이기 + 앞뒤 공백 제거)
        challenge_name = re.sub(r"\s+", " ", challenge_name.strip())
        
        # 공백만 존재하거나 5자 이하 챌린지 이름 필터링
        if not challenge_name.strip() or len(challenge_name.strip()) <= 5:
            return False, "글자 수가 너무 적어서 챌린지 생성이 불가능합니다."
        
        # 같은 단어가 3번 이상 반복된 경우 필터링 
        words = challenge_name.strip().split()
        word_counts = Counter(words)
        for word, count in word_counts.items():     # word 빼면 코드 실행 불가능 
            if count >= 3:
                return False, "같은 단어가 반복적으로 사용되어 챌린지 생성이 불가능합니다."
            
        # 같은 단어/문장으로만 이루어져있을 경우 필터링 
        if len(set(words)) == 1 and len(words) >= 3:
            return False, "동일한 단어가 반복된 챌린지 이름은 생성이 불가능합니다."
            
        # 의미가 모호한 단어가 포함되어 있는 경우 (rule-based 필터)
        ambiguous_keywords = ["그냥", "대충", "뭐든지", "뭔가", "어쩌구", "아무거나"]
        for keyword in ambiguous_keywords:
            if keyword in challenge_name:
                return False, f"'{keyword}'와 같은 의미가 모호한 단어가 포함되어 있어 챌린지 생성이 불가능합니다."
            
        # 광고성/마케팅성 문구 사전 필터링
        marketing_keywords = [
            "드려요", "경품", "추첨", "무료", "혜택", "이벤트", "지금 참여", "같이 하면", "기념", "드림", "당첨"
        ]
        for keyword in marketing_keywords:
            if keyword in challenge_name:
                return False, f"'{keyword}'와 같은 마케팅성 문구가 포함되어 있어 챌린지 생성이 불가능합니다."

        # 특수문자만 들어간 경우 사전 필터링
        if re.fullmatch(r"[^\w\s]+", challenge_name):
            return False, "특수문자만 포함된 챌린지 이름은 생성이 불가능합니다."

        # 숫자만 들어간 경우 사전 필터링 
        if challenge_name.strip().isdigit():
            return False, "숫자만 포함된 챌린지 이름은 생성이 불가능합니다."

        # 모든 기존 챌린지 이름을 나열
        existing_names = "\n".join([f"- {c.name}" for c in existing if c.name]) or "- 없음"             

        # 이름/의미 유사 여부만 판단
        prompt = (
            f"새로 생성하려는 챌린지 이름은 '{challenge_name}'입니다.\n"
            "기존 챌린지 이름 목록은 다음과 같습니다:\n"
            f"{existing_names}\n\n"

            "질문 : 위의 기존 챌린지와 유사하거나, 친환경 실천과 관련이 없는 내용인가요? \n"
            "- '예'라면 'no', 아니라면 'yes'를 출력해주세요. \n"
            "답변은 반드시 'Yes' 또는 'No' 중 하나의 단어로만 출력하세요. \n"
        )

        try:
            response = self.generate_response(prompt).strip().lower()
            print("[DEBUG] 모델 응답: \n", response)

            # 마지막 줄만 파싱
            # last_response = response.strip().split('\n')[-1].strip().lower()
            last_response = re.search(r"\b(yes|no)\b", response.strip().lower().split('\n')[-1])
            last_response = last_response.group(1) if last_response else "invalid"

            print("[DEBUG] 파싱한 단어: ", last_response)

            if last_response.startswith("no"):
                return True, "챌린지 생성이 가능합니다."
            elif last_response.startswith("yes"):
                
                for c in existing:
                    if not (c.startDate and c.endDate):
                        continue
                    try:
                        new_start = datetime.strptime(start_date, "%Y-%m-%d")
                        new_end = datetime.strptime(end_date, "%Y-%m-%d")
                        exist_start = datetime.strptime(c.startDate, "%Y-%m-%d")
                        exist_end = datetime.strptime(c.endDate, "%Y-%m-%d")

                        if dates_overlap(new_start, new_end, exist_start, exist_end):
                            return False, "동일한 챌린지가 존재하여 챌린지 생성이 불가능합니다."
                    except:
                        continue
                    
                # 'No'이지만, 기간이 겹치지 않음 
                return True, "챌린지 생성이 가능합니다."
            else:
                return False, "모델 응답이 명확하지 않아 챌린지 생성이 불가능합니다."

        except Exception as e:
            print("[BUG] 응답 실패 ", e)
            traceback.print_exc()
            return False, "챌린지 검열 중 오류가 발생했습니다."