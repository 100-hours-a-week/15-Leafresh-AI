"""
Langfuse 설정 및 CoT(Chain of Thought) 테스트를 위한 설정 파일
"""
import os
from dotenv import load_dotenv
from langfuse import Langfuse
from typing import Dict, Any, List, Optional
import json

load_dotenv()

class LangfuseConfig:
    def __init__(self):
        self.langfuse = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        )
        
    def create_trace(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """새로운 trace 생성"""
        trace = self.langfuse.trace(
            name=name,
            metadata=metadata or {}
        )
        return trace.id
    
    def log_generation(self, 
                      trace_id: str,
                      name: str,
                      model: str,
                      prompt: str,
                      completion: str,
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """LLM 생성 로깅"""
        generation = self.langfuse.generation(
            trace_id=trace_id,
            name=name,
            model=model,
            prompt=prompt,
            completion=completion,
            metadata=metadata or {}
        )
        return generation.id
    
    def log_score(self, 
                 trace_id: str,
                 name: str,
                 value: float,
                 comment: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """평가 점수 로깅"""
        self.langfuse.score(
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
        span = self.langfuse.span(
            trace_id=trace_id,
            name=name,
            input=input,
            output=output,
            metadata=metadata or {}
        )
        return span.id

# CoT 테스트를 위한 평가 함수들
class CoTEvaluator:
    def __init__(self, langfuse_config: LangfuseConfig):
        self.langfuse = langfuse_config
    
    def evaluate_thought_process(self, 
                               trace_id: str,
                               response: str,
                               expected_keywords: List[str]) -> float:
        """사고 과정 평가 - CoT 단계들이 포함되어 있는지 확인"""
        response_lower = response.lower()
        
        # CoT 관련 키워드들
        cot_keywords = [
            "생각해보면", "먼저", "그 다음", "따라서", "결론적으로",
            "분석해보면", "고려해보면", "이유는", "왜냐하면",
            "step", "단계", "과정", "사고", "추론"
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
    
    def evaluate_reasoning_steps(self,
                               trace_id: str,
                               response: str) -> float:
        """추론 단계 평가 - 단계별 사고가 명확한지"""
        # 단계 구분자들
        step_indicators = [
            "1.", "2.", "3.", "4.", "5.",
            "첫째", "둘째", "셋째", "넷째", "다섯째",
            "step 1", "step 2", "step 3",
            "단계 1", "단계 2", "단계 3"
        ]
        
        steps_found = 0
        for indicator in step_indicators:
            if indicator in response:
                steps_found += 1
        
        # 2개 이상의 단계가 있으면 좋은 점수
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
    
    def evaluate_logical_flow(self,
                            trace_id: str,
                            response: str) -> float:
        """논리적 흐름 평가"""
        # 논리적 연결어들
        logical_connectors = [
            "따라서", "그러므로", "결과적으로", "결론적으로",
            "이유는", "왜냐하면", "때문에", "그래서",
            "하지만", "그런데", "반면에", "그러나"
        ]
        
        connectors_found = 0
        for connector in logical_connectors:
            if connector in response:
                connectors_found += 1
        
        # 3개 이상의 논리적 연결어가 있으면 좋은 점수
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
    
    def evaluate_completeness(self,
                            trace_id: str,
                            response: str,
                            min_length: int = 100) -> float:
        """응답 완성도 평가"""
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