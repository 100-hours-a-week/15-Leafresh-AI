#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파인튜닝 데이터를 LangChain StructuredOutputParser 형식으로 변환
"""

import json
import os
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

def setup_parsers():
    """챌린지 추천용 ResponseSchema와 Parser 설정"""
    
    # 챌린지 추천용 스키마
    challenge_schemas = [
        ResponseSchema(
            name="recommend", 
            description="추천 텍스트를 한글로 한 문장으로 출력해주세요."
        ),
        ResponseSchema(
            name="challenges", 
            description="추천 챌린지 리스트, 각 항목은 title, description 포함"
        )
    ]
    
    challenge_parser = StructuredOutputParser.from_response_schemas(challenge_schemas)
    
    return challenge_parser

def convert_challenge_output(original_output, parser):
    """챌린지 추천 output을 LangChain 형식으로 변환"""
    
    # format_instructions 가져오기
    format_instructions = parser.get_format_instructions()
    
    # 변환된 형식 생성
    converted_output = f"""{format_instructions}

답변:
```json
{original_output}
```"""
    
    return converted_output

def convert_dataset(input_file, output_file):
    """전체 데이터셋 변환"""
    
    print(f"📂 입력 파일: {input_file}")
    print(f"📂 출력 파일: {output_file}")
    
    # Parser 설정
    challenge_parser = setup_parsers()
    
    # 원본 데이터 로드
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"📊 총 데이터 개수: {len(dataset)}")
    
    converted_dataset = []
    challenge_count = 0
    feedback_count = 0
    
    for i, item in enumerate(dataset):
        instruction = item.get('instruction', '')
        
        # 챌린지 추천 데이터인지 확인
        if '챌린지 추천 챗봇' in instruction or '챌린지' in instruction and 'JSON 형식으로 추천' in instruction:
            # 챌린지 추천 데이터 변환
            original_output = item['output']
            
            # JSON 형식인지 확인 ('{' 로 시작하는지)
            if original_output.strip().startswith('{'):
                converted_output = convert_challenge_output(original_output, challenge_parser)
                
                converted_item = {
                    "instruction": instruction,
                    "input": item['input'],
                    "output": converted_output
                }
                
                challenge_count += 1
                print(f"✅ 변환: {i+1}번 (챌린지 추천)")
            else:
                # JSON이 아닌 텍스트 형식은 그대로 유지
                converted_item = item
                print(f"⏭️  유지: {i+1}번 (텍스트 형식)")
        else:
            # 피드백 어시스턴트 등 다른 데이터는 그대로 유지
            converted_item = item
            feedback_count += 1
            print(f"⏭️  유지: {i+1}번 (피드백)")
        
        converted_dataset.append(converted_item)
    
    # 변환된 데이터 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_dataset, f, ensure_ascii=False, indent=2)
    
    print(f"\n🎉 변환 완료!")
    print(f"📈 챌린지 추천 변환: {challenge_count}개")
    print(f"📈 피드백 유지: {feedback_count}개")
    print(f"📈 전체: {len(converted_dataset)}개")
    
    return output_file

def preview_conversion():
    """변환 예시 미리보기"""
    
    print("=" * 60)
    print("🔍 변환 예시 미리보기")
    print("=" * 60)
    
    challenge_parser = setup_parsers()
    
    # 원본 예시
    original = '{"recommend": "산의 정기를 받아 더 건강하고 가벼워지는 비건 챌린지를 시작해보세요.", "challenges": [{"title": "산채비빔밥으로 점심 즐기기", "description": "주변 식당에서 신선한 나물로 만든 산채비빔밥으로 건강한 한 끼를 즐겨요."}]}'
    
    print("\n📋 원본 output:")
    print(original)
    
    # 변환된 형식
    converted = convert_challenge_output(original, challenge_parser)
    
    print(f"\n🔄 변환된 output:")
    print(converted)
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    # 미리보기 실행
    preview_conversion()
    
    # 실제 변환 실행
    input_file = "../../multitask_dataset_v3.json"
    output_file = "../../multitask_dataset_v4_langchain.json"
    
    if os.path.exists(input_file):
        convert_dataset(input_file, output_file)
    else:
        print(f"❌ 입력 파일을 찾을 수 없습니다: {input_file}") 