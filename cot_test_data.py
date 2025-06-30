"""
CoT(Chain of Thought) 테스트를 위한 테스트 데이터
"""
from typing import Dict, List, Any
from model.chatbot.chatbot_constants import label_mapping

# 거주지 및 직장 형태 정의
LOCATIONS = ["도시", "바닷가", "산", "농촌"]
WORK_TYPES = ["사무직", "현장직", "영업직", "재택"]

# 모든 조합의 base-info 테스트 케이스 생성
def generate_all_base_info_tests():
    """모든 조합의 base-info 테스트 케이스 생성"""
    test_cases = []
    test_id = 1
    
    for location in LOCATIONS:
        for work_type in WORK_TYPES:
            for category in label_mapping.keys():
                test_case = {
                    "id": f"test_{test_id:03d}",
                    "category": "base_info",
                    "input": {
                        "sessionId": f"cot_test_session_{test_id:03d}",
                        "location": location,
                        "workType": work_type,
                        "category": category
                    },
                    "expected_keywords": get_expected_keywords_for_category(category),
                    "cot_prompt": f"""
다음 사용자 정보를 바탕으로 친환경 챌린지를 추천해주세요.
사용자 정보: 위치={location}, 직업={work_type}, 관심 카테고리={category}

단계별로 생각해보세요:
1. 먼저 사용자의 상황을 분석해보세요
2. 해당 카테고리에서 실천 가능한 챌린지들을 고려해보세요
3. 사용자의 일상과 연관된 구체적인 방법을 제안해보세요
4. 최종적으로 가장 적합한 챌린지를 추천해보세요

JSON 형태로 응답해주세요:
{{
    "recommend": "추천 이유와 함께 설명",
    "challenges": [
        {{
            "title": "챌린지 제목",
            "description": "구체적인 설명"
        }}
    ]
}}
"""
                }
                test_cases.append(test_case)
                test_id += 1
    
    return test_cases

def get_expected_keywords_for_category(category: str) -> List[str]:
    """카테고리별 예상 키워드 반환"""
    category_keywords = {
        "제로웨이스트": ["제로웨이스트", "쓰레기", "일회용", "재활용", "분리수거"],
        "플로깅": ["플로깅", "조깅", "쓰레기", "운동", "환경정리"],
        "탄소발자국": ["탄소", "발자국", "이산화탄소", "기후", "온실가스"],
        "에너지 절약": ["에너지", "절약", "전기", "전력", "효율"],
        "업사이클": ["업사이클", "재활용", "창작", "디자인", "변환"],
        "문화 공유": ["문화", "공유", "미디어", "콘텐츠", "정보"],
        "디지털 탄소": ["디지털", "탄소", "인터넷", "데이터", "클라우드"],
        "비건": ["비건", "채식", "동물", "환경", "식단"]
    }
    return category_keywords.get(category, ["환경", "친환경", "지속가능"])

# Free-text 테스트 케이스들 (RAG 있음 + RAG 없음)
FREE_TEXT_TEST_CASES = [
    {
        "id": "test_129",
        "category": "free_text",
        "rag_enabled": True,  # RAG 사용
        "input": {
            "sessionId": "cot_test_session_129",
            "message": "집에서 탄소발자국을 줄이고 싶어요"
        },
        "expected_keywords": ["탄소", "발자국", "집", "줄이기", "환경"],
        "cot_prompt": """
사용자의 메시지를 분석하고 친환경 챌린지를 추천해주세요.
사용자 메시지: {message}

단계별로 생각해보세요:
1. 사용자가 원하는 것이 무엇인지 파악해보세요
2. 탄소발자국을 줄일 수 있는 방법들을 생각해보세요
3. 집에서 실천 가능한 구체적인 행동들을 정리해보세요
4. 사용자에게 가장 적합한 챌린지를 추천해보세요

JSON 형태로 응답해주세요:
{{
    "recommend": "추천 이유와 함께 설명",
    "challenges": [
        {{
            "title": "챌린지 제목",
            "description": "구체적인 설명"
        }}
    ]
}}
"""
    },
    {
        "id": "test_130",
        "category": "free_text",
        "rag_enabled": True,  # RAG 사용
        "input": {
            "sessionId": "cot_test_session_130",
            "message": "일회용품 사용을 줄이고 싶은데 어떻게 해야 할까요?"
        },
        "expected_keywords": ["일회용품", "줄이기", "재사용", "플라스틱", "환경"],
        "cot_prompt": """
사용자의 메시지를 분석하고 친환경 챌린지를 추천해주세요.
사용자 메시지: {message}

단계별로 생각해보세요:
1. 일회용품이 환경에 미치는 영향을 생각해보세요
2. 일상에서 일회용품을 대체할 수 있는 방법들을 찾아보세요
3. 구체적이고 실천 가능한 행동들을 정리해보세요
4. 사용자에게 단계별 챌린지를 제안해보세요

JSON 형태로 응답해주세요:
{{
    "recommend": "추천 이유와 함께 설명",
    "challenges": [
        {{
            "title": "챌린지 제목",
            "description": "구체적인 설명"
        }}
    ]
}}
"""
    },
    # RAG 없는 free-text 테스트 케이스들
    {
        "id": "test_131",
        "category": "free_text_no_rag",
        "rag_enabled": False,  # RAG 사용 안함
        "input": {
            "sessionId": "cot_test_session_131",
            "message": "집에서 탄소발자국을 줄이고 싶어요"
        },
        "expected_keywords": ["탄소", "발자국", "집", "줄이기", "환경"],
        "cot_prompt": """
사용자의 메시지를 분석하고 친환경 챌린지를 추천해주세요.
사용자 메시지: {message}

단계별로 생각해보세요:
1. 사용자가 원하는 것이 무엇인지 파악해보세요
2. 탄소발자국을 줄일 수 있는 방법들을 생각해보세요
3. 집에서 실천 가능한 구체적인 행동들을 정리해보세요
4. 사용자에게 가장 적합한 챌린지를 추천해보세요

JSON 형태로 응답해주세요:
{{
    "recommend": "추천 이유와 함께 설명",
    "challenges": [
        {{
            "title": "챌린지 제목",
            "description": "구체적인 설명"
        }}
    ]
}}
"""
    },
    {
        "id": "test_132",
        "category": "free_text_no_rag",
        "rag_enabled": False,  # RAG 사용 안함
        "input": {
            "sessionId": "cot_test_session_132",
            "message": "일회용품 사용을 줄이고 싶은데 어떻게 해야 할까요?"
        },
        "expected_keywords": ["일회용품", "줄이기", "재사용", "플라스틱", "환경"],
        "cot_prompt": """
사용자의 메시지를 분석하고 친환경 챌린지를 추천해주세요.
사용자 메시지: {message}

단계별로 생각해보세요:
1. 일회용품이 환경에 미치는 영향을 생각해보세요
2. 일상에서 일회용품을 대체할 수 있는 방법들을 찾아보세요
3. 구체적이고 실천 가능한 행동들을 정리해보세요
4. 사용자에게 단계별 챌린지를 제안해보세요

JSON 형태로 응답해주세요:
{{
    "recommend": "추천 이유와 함께 설명",
    "challenges": [
        {{
            "title": "챌린지 제목",
            "description": "구체적인 설명"
        }}
    ]
}}
"""
    }
]

# 모든 테스트 케이스 조합
COT_TEST_CASES = generate_all_base_info_tests() + FREE_TEXT_TEST_CASES

# CoT 평가 기준
COT_EVALUATION_CRITERIA = {
    "thought_process": {
        "weight": 0.4,
        "description": "사고 과정이 명확하게 드러나는지",
        "keywords": [
            "생각해보면", "먼저", "그 다음", "따라서", "결론적으로",
            "분석해보면", "고려해보면", "이유는", "왜냐하면"
        ]
    },
    "reasoning_steps": {
        "weight": 0.3,
        "description": "단계별 추론이 명확한지",
        "indicators": [
            "1.", "2.", "3.", "4.", "5.",
            "첫째", "둘째", "셋째", "넷째", "다섯째"
        ]
    },
    "logical_flow": {
        "weight": 0.2,
        "description": "논리적 흐름이 자연스러운지",
        "connectors": [
            "따라서", "그러므로", "결과적으로", "결론적으로",
            "이유는", "왜냐하면", "때문에", "그래서"
        ]
    },
    "completeness": {
        "weight": 0.1,
        "description": "응답이 충분히 완성되었는지",
        "min_length": 100
    }
}

def get_cot_test_case(test_id: str) -> Dict[str, Any]:
    """테스트 케이스 조회"""
    for case in COT_TEST_CASES:
        if case["id"] == test_id:
            return case
    return None

def get_all_test_cases() -> List[Dict[str, Any]]:
    """모든 테스트 케이스 조회"""
    return COT_TEST_CASES

def get_base_info_test_cases() -> List[Dict[str, Any]]:
    """base-info 테스트 케이스만 조회"""
    return [case for case in COT_TEST_CASES if case["category"] == "base_info"]

def get_free_text_test_cases() -> List[Dict[str, Any]]:
    """free-text 테스트 케이스만 조회"""
    return [case for case in COT_TEST_CASES if case["category"] == "free_text"]

def format_cot_prompt(case: Dict[str, Any]) -> str:
    """CoT 프롬프트 포맷팅"""
    if case["category"] == "base_info":
        return case["cot_prompt"].format(
            location=case["input"]["location"],
            workType=case["input"]["workType"],
            category=case["input"]["category"]
        )
    else:
        return case["cot_prompt"].format(
            message=case["input"]["message"]
        ) 