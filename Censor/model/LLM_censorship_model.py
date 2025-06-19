from vertexai import init
from vertexai.preview.generative_models import GenerativeModel

import os
from dotenv import load_dotenv
from typing import List

from datetime import datetime
from collections import Counter

class CensorshipModel :
    def __init__(self):
        load_dotenv()
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        init(project="leafresh", location="us-central1")
        self.model = GenerativeModel("gemini-2.0-flash")

    def validate(self, challenge_name: str, start_date: str, end_date: str, existing: List[dict]):
        def dates_overlap(s1, e1, s2, e2):
            return max(s1, s2) <= min(e1, e2)
        
        # 공백만 존재하거나 5자 이하 챌린지 이름 필터링
        if not challenge_name.strip() or len(challenge_name.strip()) <= 5:
            return False, "글자 수가 너무 적어서 챌린지 생성이 불가능합니다."
        
        # 같은 단어가 3번 이상 반복된 경우
        words = challenge_name.strip().split()
        word_counts = Counter(words)
        for word, count in word_counts.items():     # word 빼면 코드 실행 불가능 
            if count >= 3:
                return False, "같은 단어가 반복적으로 사용되어 챌린지 생성이 불가능합니다."
            
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

        # 모든 기존 챌린지 이름을 나열
        existing_names = "\n".join([f"- {c.name}" for c in existing if c.name]) or "- 없음"             

        # 이름/의미 유사 여부만 판단
        prompt = (
            f"새로 생성하려는 챌린지 이름은 '{challenge_name}'입니다.\n"
            "기존 챌린지 이름 목록은 다음과 같습니다:\n"
            f"{existing_names}\n\n"
            "기존 챌린지와 많은 부분 겹치는 경우에만 'No'라고 답해주세요. \n"
            "예를 들어 '텀블러 챌린지'와 '텀블러 이용 챌린지', '텀블러 사용 챌린지' 등은 모두 많은 부분이 겹치므로 'No'라고 답해주세요. \n"
            "챌린지의 이름이 환경과 관련이 없거나 의미가 모호한 경우 'No'라고 답해주세요. \n"
            "의미가 모호하거나 명확하지 않은 문장(예: 대충, 뭐든지, 그냥, 뭔가 등)이 포함되어 있으면 'No'라고 답해주세요. \n"
            "새로운 챌린지와 이름이 같거나, 단어는 다르지만 의미나 목적이 유사한 챌린지가 존재한다면 'No'라고 답해주세요.\n"
            "단어 순서가 다르거나 '매일', '도전하기', '습관화하기' 등이 추가된 경우에도 동일한 챌린지로 판단합니다. \n"
            "스팸이나 광고성, 홍보성, 마케팅성 문구일 경우 'No'라고 답해주세요. 하지만, 친환경과 관련된 경우 'Yes'라고 답해주세요. \n"
            "특수문자만 들어간 경우, 숫자만 들어간 경우 'No'라고 답해주세요. \n"
            "같은 단어가 여러번 반복되는 경우 'No'라고 답해주세요. \n"
            "그렇지 않다면 'Yes'라고 답해주세요. \n"
            "답변은 반드시 'Yes.' 또는 'No.'로 시작해야 합니다. \n"
        )

        try:
            result = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.3,
                    "top_p": 1,
                    "top_k": 32,
                    "max_output_tokens": 512
                }
            )
            answer = result.text.strip().lower()

            if answer.startswith("yes"):
                return True, "챌린지 생성이 가능합니다."
            elif answer.startswith("no"):
                
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
            return False, "챌린지 검열 중 오류가 발생했습니다."