"""
Langfuse 설정 및 CoT(Chain of Thought) 테스트를 위한 설정 파일
"""
import os
from dotenv import load_dotenv
from langfuse import Langfuse
from typing import Dict, Any, List, Optional

load_dotenv()

print("LANGFUSE_PUBLIC_KEY:", os.getenv("LANGFUSE_PUBLIC_KEY"))
print("LANGFUSE_SECRET_KEY:", os.getenv("LANGFUSE_SECRET_KEY"))
print("LANGFUSE_HOST:", os.getenv("LANGFUSE_HOST"))
class LangfuseConfig:
    def __init__(self):
        self.langfuse = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST")
        )
        print("self.langfuse instance:", type(self.langfuse))
    
    def create_trace_id(self) -> str:
        return self.langfuse.create_trace_id()
    
    def log_generation(self, trace_id: str, name: str, model: str, prompt: str, completion: str, metadata=None) -> str:
        # trace_id로 generation 생성
        generation = self.langfuse.create_generation(
            trace_id=trace_id,
            name=name,
            model=model,
            prompt=prompt,
            completion=completion,
            metadata=metadata or {}
        )
        return generation.id
    
    def log_score(self, trace_id: str, name: str, value: float, comment: str = None, metadata=None):
        # trace_id로 score 생성
        self.langfuse.create_score(
            trace_id=trace_id,
            name=name,
            value=value,
            comment=comment,
            metadata=metadata or {}
        )
    
    def log_span(self,
                trace_id: str,
                name: str,
                input: Optional[Dict[str, Any]] = None,
                output: Optional[Dict[str, Any]] = None,
                metadata: Optional[Dict[str, Any]] = None) -> str:
        """Span 로깅 (함수 호출 등)"""
        # trace_id로 span 생성
        span = self.langfuse.create_span(
            trace_id=trace_id,
            name=name,
            input=input,
            output=output,
            metadata=metadata or {}
        )
        return span.id

# CoT 테스트를 위한 평가 함수들
"""사고 과정 평가 - CoT 단계들이 포함되어 있는지 확인"""
class CoTEvaluator:
    def __init__(self, langfuse_config: LangfuseConfig):
        self.langfuse = langfuse_config
    
    def evaluate_thought_process(self, 
                               trace_id: str,
                               response: str,
                               expected_keywords: List[str]) -> float:
        response_lower = response.lower()
        
        # CoT 관련 키워드들 (확장)
        cot_keywords = [
            # 사고 과정 키워드
            "생각해보면", "먼저", "그 다음", "따라서", "결론적으로",
            "분석해보면", "고려해보면", "이유는", "왜냐하면",
            "step", "단계", "과정", "사고", "추론",
            # 추가 키워드 (더 일반적인 표현들)
            "이런", "이러한", "이런 식으로", "이렇게", "이런 방식으로",
            "고민해보면", "살펴보면", "확인해보면", "점검해보면",
            "검토해보면", "살펴보면", "살펴보면", "살펴보면",
            # 더 일반적인 한국어 표현들
            "추천", "제안", "방법", "방안", "해결책",
            "효과적", "효율적", "적합한", "좋은", "나쁜",
            "환경", "친환경", "지속가능", "탄소", "에너지",
            "실천", "실행", "적용", "활용", "이용"
        ]
        
        # CoT 키워드 포함 여부
        cot_score = 0.0
        found_cot_keywords = []
        for keyword in cot_keywords:
            if keyword in response_lower:
                cot_score += 0.1
                found_cot_keywords.append(keyword)
        
        cot_score = min(cot_score, 1.0)  # 최대 1.0
        
        # 예상 키워드 포함 여부
        keyword_score = 0.0
        found_expected_keywords = []
        for keyword in expected_keywords:
            if keyword.lower() in response_lower:
                keyword_score += 1.0 / len(expected_keywords)
                found_expected_keywords.append(keyword)
        
        # 최종 점수 (CoT 60%, 키워드 40%)
        final_score = (cot_score * 0.6) + (keyword_score * 0.4)
        
        self.langfuse.log_score(
            trace_id=trace_id,
            name="cot_thought_process",
            value=final_score,
            comment=f"CoT keywords found: {found_cot_keywords} | Expected keywords: {found_expected_keywords}",
            metadata={
                "cot_score": cot_score,
                "keyword_score": keyword_score,
                "found_cot_keywords": found_cot_keywords,
                "found_expected_keywords": found_expected_keywords,
                "expected_keywords": expected_keywords
            }
        )
        
        return final_score
    
    """추론 단계 평가 - 단계별 사고가 명확한지"""
    def evaluate_reasoning_steps(self,
                               trace_id: str,
                               response: str) -> float:
        # 단계 구분자들 (확장)
        step_indicators = [
            # 숫자 기반
            "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "10.",
            "1)", "2)", "3)", "4)", "5)",
            # 한글 기반
            "첫째", "둘째", "셋째", "넷째", "다섯째", "여섯째", "일곱째", "여덟째",
            "첫 번째", "두 번째", "세 번째", "네 번째", "다섯 번째",
            # 영어 기반
            "step 1", "step 2", "step 3", "step 4", "step 5",
            "first", "second", "third", "fourth", "fifth",
            # 단계 표현
            "단계 1", "단계 2", "단계 3", "단계 4", "단계 5",
            "1단계", "2단계", "3단계", "4단계", "5단계",
            # 기타 표현
            "우선", "다음으로", "그 다음", "마지막으로", "결론적으로",
            "먼저", "그리고", "또한", "또한", "또한",
            # JSON 응답에서 자주 나오는 표현들
            "분석", "검토", "선별", "제안", "추천",
            "고려", "검토", "평가", "선택", "결정"
        ]
        
        steps_found = 0
        for indicator in step_indicators:
            if indicator in response:
                steps_found += 1
        
        # 1개 이상의 단계가 있으면 기본 점수, 2개 이상이면 좋은 점수
        if steps_found == 0:
            score = 0.0
        elif steps_found == 1:
            score = 0.3
        else:
            score = min(steps_found / 3.0, 1.0)
        
        self.langfuse.log_score(
            trace_id=trace_id,
            name="reasoning_steps",
            value=score,
            comment=f"Found {steps_found} reasoning steps",
            metadata={
                "steps_found": steps_found,
                "response_length": len(response)
            }
        )
        
        return score
    
    """논리적 흐름 평가"""
    def evaluate_logical_flow(self,
                            trace_id: str,
                            response: str) -> float:
        # 논리적 연결어들 (확장)
        logical_connectors = [
            # 결론 연결어
            "따라서", "그러므로", "결과적으로", "결론적으로", "그래서",
            # 이유 연결어
            "이유는", "왜냐하면", "때문에", "~하기 때문에", "~이므로",
            # 대조 연결어
            "하지만", "그런데", "반면에", "그러나", "다만", "단",
            # 추가 연결어
            "또한", "또한", "그리고", "또한", "또한",
            # 순서 연결어
            "먼저", "그 다음", "그리고", "마지막으로", "결국",
            # 기타 논리적 표현
            "이런 식으로", "이렇게", "이런 방식으로", "이런 이유로",
            "이런 관점에서", "이런 측면에서", "이런 의미에서"
        ]
        
        connectors_found = 0
        for connector in logical_connectors:
            if connector in response:
                connectors_found += 1
        
        # 1개 이상의 논리적 연결어가 있으면 기본 점수, 2개 이상이면 좋은 점수
        if connectors_found == 0:
            score = 0.0
        elif connectors_found == 1:
            score = 0.3
        else:
            score = min(connectors_found / 4.0, 1.0)
        
        self.langfuse.log_score(
            trace_id=trace_id,
            name="logical_flow",
            value=score,
            comment=f"Found {connectors_found} logical connectors",
            metadata={
                "connectors_found": connectors_found
            }
        )
        
        return score
    
    """응답 완성도 평가"""
    def evaluate_completeness(self,
                            trace_id: str,
                            response: str,
                            min_length: int = 100) -> float:
        length = len(response)
        
        if length < min_length:
            score = length / min_length
        else:
            score = 1.0
        
        self.langfuse.log_score(
            trace_id=trace_id,
            name="response_completeness",
            value=score,
            comment=f"Response length: {length} characters",
            metadata={
                "actual_length": length,
                "min_length": min_length
            }
        )
        
        return score

# 전역 인스턴스
langfuse_config = LangfuseConfig()
cot_evaluator = CoTEvaluator(langfuse_config) 