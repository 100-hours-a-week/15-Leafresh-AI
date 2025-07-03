import os, traceback, re
from dotenv import load_dotenv
from typing import List

from datetime import datetime
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import asyncio

# GCP model_dir = "/home/ubuntu/hyperclovax_model/models--naver-hyperclovax--HyperCLOVAX-SEED-Text-Instruct-1.5B/snapshots/543a1be9d6233069842ffce73aa56a232a4f457b"
# local model_dir = "./hyperclovax_model/models--naver-hyperclovax--HyperCLOVAX-SEED-Text-Instruct-1.5B/snapshots/543a1be9d6233069842ffce73aa56a232a4f457b"
class HyperClovaxModel :
    def __init__(self, model_dir = "./hyperclovax_model/models--naver-hyperclovax--HyperCLOVAX-SEED-Text-Instruct-1.5B/snapshots/543a1be9d6233069842ffce73aa56a232a4f457b"):
        load_dotenv()
        self.device = "cpu"
        self.semaphore = asyncio.Semaphore(2)  

        print("[INFO] HyperCLOVA-X 모델 로딩 중 ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float32,                          # CPU 환경에서는 float16 사용 불가
        ).to(self.device)
        print("[INFO] 모델 로딩 완료")


    async def generate_response(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=64,
                temperature=None,
                top_p=1.0,
                # top_k=32,
                eos_token_id=self.tokenizer.eos_token_id, # 종료 토큰 지정 < /s>
                pad_token_id=self.tokenizer.eos_token_id
            )

        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result.strip()
    

    async def validate(self, challenge_name: str, start_date: str, end_date: str, existing: List[dict]):
        # 날짜 계산 함수 
        def dates_overlap(s1, e1, s2, e2):
            return max(s1, s2) <= min(e1, e2)
        
        # 유사도 필터 추가
        def has_keyword_overlap(name1, name2, threshold=1):
            """두 챌린지 이름 간 중복 핵심 단어 수를 확인 (불용어 제외)"""
            stopwords = {
                "챌린지", "잘", "하기", "이용", "사용", "운동", "습관", "실천", "활동", "함께", "나누기", "참여"
            }

            # 단어 추출 후 소문자 처리 및 stopword 제거
            set1 = {w.lower() for w in re.findall(r'\w+', name1) if w not in stopwords}
            set2 = {w.lower() for w in re.findall(r'\w+', name2) if w not in stopwords}

            overlap = set1 & set2
            return len(overlap) >= threshold

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
        ambiguous_keywords = [
            "그냥", "대충", "뭐든지", "뭐하지", "뭔가", "어쩌구", "아무거나", "이런저런", "대강", "등등", "기타", "뭐시기", "뭐더라", "뭐였지", "이거저거", "이것저것", "뭐라고", "어쩌다", "마음대로", "알아서", "적절히", "어느정도", "그런거", "이런거"
        ]
        for keyword in ambiguous_keywords:
            if keyword in challenge_name:
                return False, f"'{keyword}'와 같은 의미가 모호한 단어가 포함되어 있어 챌린지 생성이 불가능합니다."
            
        # 광고성/마케팅성 문구 사전 필터링
        marketing_keywords = [
            "드려요", "드림", "경품", "추첨", "무료", "혜택", "이벤트", "지금 참여", "같이 하면", "기념", "당첨", "광고", "특가", "세일", "할인", "사은품", "증정", "한정", "핫딜", "초특가", "파격"
        ]
        for keyword in marketing_keywords:
            if keyword in challenge_name:
                return False, f"'{keyword}'와 같은 마케팅성 문구가 포함되어 있어 챌린지 생성이 불가능합니다."

        # 비속어 사전 필터링
        profanity_keywords = [
            "ㅆㅂ", "ㅅㅂ", "ㅂㅅ", "ㄱㅅ", "ㅄ", "씨발", "시발", "개새", "존나", "좆", "fuck", "shit", "bitch", "fuckyou", "asshole", "motherfucker", "죽어", "꺼져", "엿먹어", "니애미", "니애비", "개같", "멍청이", "병신", "씨XX"
        ]
        for keyword in profanity_keywords:
            if keyword in challenge_name:
                return False, f"'{keyword}'와 같은 비속어 또는 부적절한 표현이 포함되어 있어 챌린지 생성이 불가능합니다."

        # 특수문자만 들어간 경우 사전 필터링
        if re.fullmatch(r"[^\w\s]+", challenge_name) or re.fullmatch(r"[_\W]+", challenge_name):
            return False, "특수문자만 포함된 챌린지 이름은 생성이 불가능합니다."
        
        # 숫자만 들어간 경우 사전 필터링 
        if challenge_name.strip().isdigit():
            return False, "숫자만 포함된 챌린지 이름은 생성이 불가능합니다."  

        # 영어만 포함된 챌린지 이름 필터링
        if re.fullmatch(r"[A-Za-z\s]+", challenge_name):
            return False, "영문으로만 작성된 챌린지 이름은 생성이 불가능합니다."       

        # 친환경 여부 판단 
        prompt_first = (
            f"질문: '{challenge_name}' 챌린지는 친환경 실천과 관련이 있나요? \n\n"
            
            "답변은 반드시 'yes' 또는 'no' 중 하나만 출력하세요. \n"
            "다른 말은 절대 하지 마세요. \n"
        )

        response_first = (await self.generate_response(prompt_first)).strip().lower()
        print("[DEBUG] 첫번째 모델 응답: ", response_first, "\n")

        match_first = re.search(r"\b(yes|no|네|예|아니오|아니요)\b", response_first.strip().lower().split('\n')[-1])
        
        if match_first:
            keyword = match_first.group(1)
            if keyword in ["yes", "네", "예"]:
                last_response_first = "yes"
            elif keyword in ["no", "아니오", "아니요"]:
                last_response_first = "no"
            else:
                last_response_first = "invalid"
        else:
            last_response_first = "invalid"
        print("[DEBUG] 첫번째 모델 응답 파싱: ", last_response_first, "\n")

        if last_response_first == "no":
            return False, "친환경 실천과 관련 없는 챌린지는 생성할 수 없습니다."
        
        # 날짜 겹치는 챌린지만 필터링
        conflicting = []

        try:
            new_start = datetime.strptime(start_date, "%Y-%m-%d")
            new_end = datetime.strptime(end_date, "%Y-%m-%d")

            for c in existing:
                if not (c.startDate and c.endDate):
                    continue
                exist_start = datetime.strptime(c.startDate, "%Y-%m-%d")
                exist_end = datetime.strptime(c.endDate, "%Y-%m-%d")

                if dates_overlap(new_start, new_end, exist_start, exist_end):
                    conflicting.append(c)
        except:
            pass
        
        if not conflicting:
            print("[DEBUG] 날짜가 겹치는 챌린지가 존재하지 않아 챌린지 생성 가능 ")
            return True, "기간이 겹치는 기존 챌린지가 없으므로 생성 가능합니다."
        
        # 겹치는 챌린지 이름 추출 
        existing_names = "\n".join([f"- {c.name}" for c in conflicting if c.name]) or "- 없음"    

        # 이름/의미 유사 여부만 판단
        prompt_second = (
            f"새로 생성하려는 챌린지 이름은 '{challenge_name}'입니다.\n"
            "기존 챌린지 이름 목록은 다음과 같습니다:\n"
            f"{existing_names}\n\n"

            "질문 : 새로 생성하려는 챌린지와 매우 비슷하거나 동일한 기존의 챌린지가 목록에 존재하나요? \n"
            "답변은 반드시 'yes' 또는 'no' 중 하나만 출력하세요. \n"
            "다른 말은 절대 하지 마세요. \n"
        )

        try:
            response_second = (await self.generate_response(prompt_second)).strip().lower()
            print("[DEBUG] 두번째 모델 응답: ", response_second, "\n")

            # 마지막 줄만 파싱
            # last_response = response.strip().split('\n')[-1].strip().lower()
            match_second = re.search(r"\b(yes|no|네|예|아니오|아니요)\b", response_second.strip().lower().split('\n')[-1])

            if match_second:
                last_response_second = match_second.group(1)

                if last_response_second in ["no", "아니오", "아니요"]:
                    print("[DEBUG] 두번째 모델 응답 파싱: no (생성 가능)")
                    return True, "의미상 유사한 챌린지가 없으므로 생성 가능합니다."
                
                elif last_response_second in ["yes", "네", "예"]:
                    print("[DEBUG] 두번째 모델 응답 파싱: yes (생성 불가능)")

                    has_overlap = any(has_keyword_overlap(challenge_name, c.name, threshold=1) for c in conflicting)
                    if not has_overlap:
                        print("[DEBUG] 유사 키워드 없음 → yes 판단 무효 → 생성 가능")
                        return True, "실제 유사 단어가 없으므로 생성 가능합니다."
                    
                    return False, "유사하고 기간이 겹치는 챌린지가 존재하여 생성이 불가능합니다."
                    
                else:
                    print("[DEBUG] 두번째 모델 응답 파싱: invalid")
                    return False, "모델 응답이 명확하지 않아 챌린지 생성이 불가능합니다."
                
            else:
                print("[DEBUG] 두번째 모델 응답 파싱: invalid")
                return False, "모델 응답이 명확하지 않아 챌린지 생성이 불가능합니다."

        except Exception as e:
            print("[BUG] 응답 실패 ", e)
            traceback.print_exc()
            return False, "챌린지 검열 중 오류가 발생했습니다."
        