"""
CoT(Chain of Thought) 테스트 실행 스크립트
"""
import asyncio
import json
import time
from typing import Dict, Any, List
import httpx
from langfuse_config import langfuse_config, cot_evaluator
from cot_test_data import get_all_test_cases, format_cot_prompt

class CoTTestRunner:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def run_base_info_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """기본 정보 기반 테스트 실행"""
        trace_id = langfuse_config.create_trace(
            name=f"CoT_BaseInfo_Test_{test_case['id']}",
            metadata={
                "test_id": test_case["id"],
                "category": test_case["category"],
                "input": test_case["input"]
            }
        )
        
        try:
            # API 호출
            response = await self.client.post(
                f"{self.base_url}/ai/chatbot/recommendation/base-info",
                json=test_case["input"]
            )
            
            if response.status_code == 200:
                result = response.json()
                recommendation = result.get("data", {}).get("recommend", "")
                
                # CoT 평가 실행
                scores = await self.evaluate_cot_response(
                    trace_id, recommendation, test_case["expected_keywords"]
                )
                
                return {
                    "test_id": test_case["id"],
                    "status": "success",
                    "trace_id": trace_id,
                    "response": recommendation,
                    "scores": scores,
                    "api_response": result
                }
            else:
                return {
                    "test_id": test_case["id"],
                    "status": "error",
                    "trace_id": trace_id,
                    "error": f"API Error: {response.status_code}",
                    "response_text": response.text
                }
                
        except Exception as e:
            return {
                "test_id": test_case["id"],
                "status": "error",
                "trace_id": trace_id,
                "error": str(e)
            }
    
    async def run_free_text_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """자유 텍스트 기반 테스트 실행"""
        trace_id = langfuse_config.create_trace(
            name=f"CoT_FreeText_Test_{test_case['id']}",
            metadata={
                "test_id": test_case["id"],
                "category": test_case["category"],
                "input": test_case["input"]
            }
        )
        
        try:
            # API 호출
            response = await self.client.post(
                f"{self.base_url}/ai/chatbot/recommendation/free-text",
                json=test_case["input"]
            )
            
            if response.status_code == 200:
                result = response.json()
                recommendation = result.get("data", {}).get("recommend", "")
                
                # CoT 평가 실행
                scores = await self.evaluate_cot_response(
                    trace_id, recommendation, test_case["expected_keywords"]
                )
                
                return {
                    "test_id": test_case["id"],
                    "status": "success",
                    "trace_id": trace_id,
                    "response": recommendation,
                    "scores": scores,
                    "api_response": result
                }
            else:
                return {
                    "test_id": test_case["id"],
                    "status": "error",
                    "trace_id": trace_id,
                    "error": f"API Error: {response.status_code}",
                    "response_text": response.text
                }
                
        except Exception as e:
            return {
                "test_id": test_case["id"],
                "status": "error",
                "trace_id": trace_id,
                "error": str(e)
            }
    
    async def evaluate_cot_response(self, trace_id: str, response: str, expected_keywords: List[str]) -> Dict[str, float]:
        """CoT 응답 평가"""
        scores = {}
        
        # 사고 과정 평가
        scores["thought_process"] = cot_evaluator.evaluate_thought_process(
            trace_id, response, expected_keywords
        )
        
        # 추론 단계 평가
        scores["reasoning_steps"] = cot_evaluator.evaluate_reasoning_steps(
            trace_id, response
        )
        
        # 논리적 흐름 평가
        scores["logical_flow"] = cot_evaluator.evaluate_logical_flow(
            trace_id, response
        )
        
        # 완성도 평가
        scores["completeness"] = cot_evaluator.evaluate_completeness(
            trace_id, response
        )
        
        # 종합 점수 계산
        weights = {
            "thought_process": 0.4,
            "reasoning_steps": 0.3,
            "logical_flow": 0.2,
            "completeness": 0.1
        }
        
        total_score = sum(scores[key] * weights[key] for key in scores)
        scores["total"] = total_score
        
        # 종합 점수 로깅
        langfuse_config.log_score(
            trace_id=trace_id,
            name="cot_total_score",
            value=total_score,
            comment=f"Total CoT Score: {total_score:.3f}",
            metadata={
                "individual_scores": scores,
                "weights": weights
            }
        )
        
        return scores
    
    async def run_all_tests(self) -> List[Dict[str, Any]]:
        """모든 테스트 실행"""
        test_cases = get_all_test_cases()
        results = []
        
        print(f"🚀 CoT 테스트 시작: {len(test_cases)}개 테스트 케이스")
        print("=" * 50)
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"📝 테스트 {i}/{len(test_cases)}: {test_case['id']}")
            
            if test_case["category"] == "base_info":
                result = await self.run_base_info_test(test_case)
            else:
                result = await self.run_free_text_test(test_case)
            
            results.append(result)
            
            # 결과 출력
            if result["status"] == "success":
                scores = result["scores"]
                print(f"✅ 성공 - 총점: {scores['total']:.3f}")
                print(f"   사고과정: {scores['thought_process']:.3f}")
                print(f"   추론단계: {scores['reasoning_steps']:.3f}")
                print(f"   논리흐름: {scores['logical_flow']:.3f}")
                print(f"   완성도: {scores['completeness']:.3f}")
            else:
                print(f"❌ 실패: {result.get('error', 'Unknown error')}")
            
            print(f"   Trace ID: {result['trace_id']}")
            print("-" * 30)
            
            # 테스트 간 간격
            await asyncio.sleep(1)
        
        await self.client.aclose()
        return results
    
    def generate_report(self, results: List[Dict[str, Any]]) -> str:
        """테스트 결과 리포트 생성"""
        successful_tests = [r for r in results if r["status"] == "success"]
        failed_tests = [r for r in results if r["status"] == "error"]
        
        report = f"""
# CoT 테스트 결과 리포트

## 📊 전체 통계
- 총 테스트: {len(results)}개
- 성공: {len(successful_tests)}개
- 실패: {len(failed_tests)}개
- 성공률: {len(successful_tests)/len(results)*100:.1f}%

## 🎯 성공한 테스트 상세 결과
"""
        
        if successful_tests:
            total_scores = [r["scores"]["total"] for r in successful_tests]
            avg_score = sum(total_scores) / len(total_scores)
            
            report += f"""
### 평균 점수: {avg_score:.3f}

| 테스트 ID | 총점 | 사고과정 | 추론단계 | 논리흐름 | 완성도 | Trace ID |
|-----------|------|----------|----------|----------|--------|----------|
"""
            
            for result in successful_tests:
                scores = result["scores"]
                report += f"| {result['test_id']} | {scores['total']:.3f} | {scores['thought_process']:.3f} | {scores['reasoning_steps']:.3f} | {scores['logical_flow']:.3f} | {scores['completeness']:.3f} | {result['trace_id']} |\n"
        
        if failed_tests:
            report += f"""
## ❌ 실패한 테스트
"""
            for result in failed_tests:
                report += f"- {result['test_id']}: {result.get('error', 'Unknown error')}\n"
        
        report += f"""
## 🔗 Langfuse 대시보드
테스트 결과를 자세히 보려면 Langfuse 대시보드를 확인하세요.
각 테스트의 Trace ID를 사용하여 개별 결과를 조회할 수 있습니다.
"""
        
        return report

async def main():
    """메인 실행 함수"""
    runner = CoTTestRunner()
    
    try:
        results = await runner.run_all_tests()
        report = runner.generate_report(results)
        
        # 리포트 저장
        with open("cot_test_report.md", "w", encoding="utf-8") as f:
            f.write(report)
        
        print("\n" + "=" * 50)
        print("📋 테스트 완료! 결과가 cot_test_report.md에 저장되었습니다.")
        print("=" * 50)
        print(report)
        
    except Exception as e:
        print(f"❌ 테스트 실행 중 오류 발생: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 