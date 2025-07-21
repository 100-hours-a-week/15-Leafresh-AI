# LLM 파인튜닝 트러블슈팅 (Troubleshooting Fine-tuning)

data[25-07-15]

## 1. Loss가 잘 안 줄어드는 경우
- **데이터셋이 너무 작거나 중복/유사 데이터가 많음**
  - 더 많은 데이터, 다양한 상황/문장/피드백 패턴 추가
- **output이 너무 짧거나 단순**
  - 다양한 문장 구조, 이모지, 피드백 스타일 추가
- **output이 항상 비슷한 패턴**
  - recommend/칭찬 멘트 pool, challenge title/description pool 확장
- **하이퍼파라미터 문제**
  - learning rate, batch size, optimizer, scheduler 등 조정

## 2. 데이터 품질 문제
- **output이 JSON 형식이 아니거나, 필드 누락**
  - output 구조 일관성 체크 (recommend, challenges, title, description 등)
- **피드백 output이 100자 초과**
  - 길이 제한 로직 추가, 자연스러운 축약/요약
- **이모지/문장 다양성 부족**
  - 다양한 이모지, 문장 템플릿 활용

## 3. 모델/토크나이저 이슈
- **토크나이저가 특수문자/이모지 처리 못함**
  - 토크나이저 vocab 확장, pre-tokenization 확인
- **output이 깨지거나 잘림**
  - max_length, truncation 등 파라미터 확인

## 4. 기타
- **학습이 너무 빨리 끝남/과적합**
  - early stopping, validation split, regularization 적용
- **실제 inference 시 output이 기대와 다름**
  - instruction/input/output 예시 다양화, prompt engineering

---

> **실전 팁**
> - 데이터셋을 늘릴 때는 기존 output pool에서 변형/조합, 랜덤 샘플링, 템플릿 변형을 적극 활용
> - 피드백 output은 100자 이내, 나머지는 기존 패턴 유지
> - loss가 plateau에 머물면 데이터/하이퍼파라미터/모델 구조 모두 점검 

data[25-07-17]

# 25-07-17 토크나이저 config 및 vLLM 관련 트러블슈팅

- 이 섹션은 2025-07-17에 발생한 토크나이저 및 vLLM 관련 문제와 해결 과정을 상세히 기록합니다.

## 1. vLLM에서 TypeError: expected str, bytes or os.PathLike object, not NoneType 에러 발생

### 문제점
- vLLM 실행 시 토크나이저 로딩에서 TypeError 발생

### 원인 분석
- tokenizer_config.json에 "tokenizer_file" 또는 "vocab_file" 항목이 없어서, transformers/vLLM이 tokenizer.model 파일을 자동으로 못 찾음

### 해결책
- tokenizer_config.json에 반드시 "tokenizer_file": "tokenizer.model" 또는 "vocab_file": "tokenizer.model" 추가

---

## 2. eos_token 값이 " "(공백)으로 잘못 설정된 경우

### 문제점
- 파인튜닝/수동 config 수정 시 실수로 "eos_token": " "로 바뀜

### 원인 분석
- 문장 종료가 제대로 인식되지 않아 출력이 비정상적으로 길어지거나, 디코더가 오작동

### 해결책
- 반드시 "eos_token": "</s>"로 설정

---

## 3. 원본 모델은 config에 tokenizer_file이 없어도 동작하는데, 파인튜닝/수동 모델은 왜 필요?

### 문제점
- 원본 모델은 config에 tokenizer_file이 없어도 동작, 파인튜닝/수동 모델은 에러 발생

### 원인 분석
- 원본 모델은 Hugging Face 표준 구조라 transformers가 자동 인식, 파인튜닝/수동 복사 모델은 구조가 미묘하게 다를 수 있어 명시 필요

### 해결책/팁
- 파인튜닝/수동 모델은 config에 명시적으로 경로를 적어주는 것이 가장 안전함

---

data[25-07-18]

# 25-07-18 vLLM device-side assert 및 파인튜닝 품질 문제 트러블슈팅

- 이 섹션은 2025-07-18에 발생한 vLLM device-side assert 문제와 파인튜닝 결과 품질 문제를 상세히 기록합니다.

## 1. vLLM에서 CUDA device-side assert triggered 에러 발생

### 문제점
- vLLM 서버는 정상 기동되지만, inference 요청 시 "CUDA error: device-side assert triggered" 에러 발생
- transformers에서는 정상 동작하지만 vLLM에서만 에러 발생

### 원인 분석
- **모델/토크나이저/config 파일 mismatch**: merge된 모델의 config, tokenizer 파일이 base 모델과 다름
- **vLLM의 엄격한 GPU 커널 체크**: transformers는 fallback이 있지만 vLLM은 GPU에서 엄격하게 체크
- **LoRA merge 후 config/tokenizer 파일 누락**: merge_and_unload()는 weight만 합치고, config/tokenizer는 별도로 base와 동일하게 맞춰야 함

### 해결책
1. **base 모델의 config/tokenizer 파일을 merge된 모델에 덮어쓰기**
   - config.json, tokenizer_config.json, generation_config.json
   - tokenizer.model, tokenizer.json, special_tokens_map.json
2. **tokenizer_config.json에 "tokenizer_file": "tokenizer.model" 추가**
3. **모든 파일이 base 모델과 완전히 동일한지 확인**

### 핵심 원리
- **LoRA merge는 weight만 합치는 것**: 구조(config), 토크나이저는 base와 동일해야 함
- **vLLM은 GPU 커널에서 엄격하게 체크**: config/tokenizer mismatch가 있으면 바로 device-side assert 발생

---

## 2. 파인튜닝 결과 품질 문제 (한글 토큰화, 이상한 출력)

### 문제점
- 출력이 이상함: "힐링할수있는시는건강!" (띄어쓰기 없음)
- 이상한 토큰: "Band", "공HED", "abandon", "anonymousanimal" 등
- 문맥 부족: 제대로 된 챌린지가 아닌 이상한 텍스트

### 원인 분석
- **한글 토큰화 문제**: 토크나이저가 한글 띄어쓰기를 제대로 처리하지 못함
- **데이터셋 품질**: 학습 데이터의 한글 띄어쓰기, 문장 구조가 부족
- **파인튜닝 파라미터**: learning rate, epochs, batch size 등이 최적화되지 않음

### 해결책
1. **데이터셋 품질 개선**
   - 한글 띄어쓰기 정확히: "힐링할 수 있는 시는 건강!"
   - 더 구체적이고 현실적인 챌린지 예시 추가
   - JSON 형식 엄격하게 지키기
2. **파인튜닝 파라미터 조정**
   - learning_rate: 더 작게 (예: 1e-4 → 5e-5)
   - epochs: 더 많이 (예: 3 → 5-10)
   - batch_size: 메모리 허용시 더 크게
3. **프롬프트 개선**
   - 더 명확한 instruction
   - 예시 포함 (few-shot learning)

---

## 3. 왜 merge된 모델의 config/tokenizer가 base와 동일해야 하나?

### 핵심 원리
- **LoRA는 "가중치만" 수정하는 방식**: 모델 구조(config), 토크나이저는 그대로 유지
- **merge_and_unload()는 weight만 base에 합치는 것**: config/tokenizer는 별도로 base와 동일하게 맞춰야 함
- **vLLM의 엄격한 요구사항**: GPU 커널에서 정확한 구조 정보를 요구, config.json의 vocab_size, hidden_size, num_attention_heads 등이 weight 파일과 1:1로 일치해야 함

### transformers vs vLLM 차이
- **transformers**: 내부적으로 fallback/에러 무시 가능
- **vLLM**: GPU 커널에서 엄격하게 체크 → 조금이라도 다르면 바로 device-side assert 발생

---

## 현재 상태
- vLLM device-side assert 문제는 해결됨
- 파인튜닝 결과 품질 개선 필요 (한글 토큰화, 데이터셋 품질, 파라미터 튜닝)
- config/tokenizer를 base와 동일하게 맞추는 것이 핵심임을 확인 

data[25-07-20]

# 25-07-20 데이터셋 포맷 다양성(줄바꿈/JSON) 혼합 이슈 (최신 코드 반영)

## 문제점 (이전)
- 자유채팅/챌린지 추천 데이터셋의 output 포맷이 줄바꿈(\n) 문자열과 JSON 문자열이 혼합되어 있었음
- 파인튜닝/추론 시 모델이 다양한 포맷(줄바꿈/JSON)으로 답변을 생성함
- 실제 서비스에서 output 포맷이 일관되지 않아 후처리/파싱 로직이 복잡해질 수 있었음

## 원인 분석 (이전)
- 데이터 생성 코드(`normalize_and_generate_multitask_dataset.py`)에서 챌린지 추천 output 포맷을 랜덤(50%)으로 줄바꿈 문자열 또는 JSON 문자열로 생성하도록 설계함
- 목적: 데이터 다양성, 실제 서비스에서의 포맷 적응력 향상, 프롬프트/실전 상황에서 모델이 여러 포맷을 유연하게 생성할 수 있도록 하기 위함
- 하지만, 한 포맷만 원하는 경우에는 혼합이 오히려 불편할 수 있었음

## 해결책 (최신)
- **2025-07-20 코드 변경:** 챌린지 추천 데이터의 output 포맷을 무조건 줄바꿈(\n) 문자열로만 생성하도록 코드 수정
  - `generate_challenge_items` 함수에서 if문/else문 제거, 항상 줄바꿈 포맷만 생성
  - 예시:
    ```python
    challenge_lines = [f"{ch['title']}: {ch['description']}" for ch in challenges]
    output_text = recommend + "\n" + "\n".join(challenge_lines)
    ```
- 최종 저장 파일명도 `multitask_dataset_final.json` → `multitask_dataset_v2.json`으로 변경
- 자유채팅(프리텍스트) 데이터도 이미 줄바꿈 포맷만 생성하도록 유지
- 데이터셋 생성 시 output 포맷을 일관되게 맞추면 모델이 해당 포맷을 더 잘 따라함
- 서비스 요구사항에 따라 포맷 일관성(파싱/후처리 단순화) 또는 다양성(모델 적응력 강화) 중 선택 가능

## 실전 사례/운영 경험 (최신)
- **실제 운영 중**: output 포맷이 혼합되어 있을 때, 프론트엔드/백엔드에서 파싱 로직이 복잡해지고, 예외 케이스(예: 줄바꿈 포맷인데 JSON 파싱 시도, JSON인데 줄바꿈 split 시도 등)에서 오류가 발생할 수 있었음
- **코드 수정 후**: 챌린지 추천/프리텍스트 output이 모두 줄바꿈 포맷으로 통일되어, 후처리/파싱 로직이 단순해지고, 운영 안정성 향상
- **실전 팁**: 데이터셋을 늘릴 때는 기존 output pool에서 변형/조합, 랜덤 샘플링, 템플릿 변형을 적극 활용하고, 후처리 로직을 반드시 준비할 것

## 현재 상태 (2025-07-20 기준)
- 챌린지 추천/프리텍스트 데이터의 output이 모두 줄바꿈(\n) 포맷으로 통일됨
- 최종 데이터셋 파일: `multitask_dataset_v2.json`
- 모델이 일관된 포맷을 학습/생성하므로 서비스 후처리/파싱이 매우 단순해짐
- 필요시 포맷 다양성(혼합)도 코드 수정으로 쉽게 적용 가능

--- 

# 2025-07-21 챗봇 프롬프트 및 라우터 코드 개선 내역

## 1. 챗봇 프롬프트(LLM_chatbot_base_info_model.py, LLM_chatbot_free_text_model.py) 변경
- 프롬프트를 **한글 100%**로, recommend/challenges 등 모든 출력이 반드시 한글로만 나오도록 강하게 명시
- 반드시 하나의 올바른 JSON 객체만 출력, recommend(문자열)와 challenges(객체 배열)만 최상위 필드로
- challenges 각 항목의 title/description도 한글로만, 영어/숫자/특수문자/이모지/마크다운/코드블록 등 사용 금지
- JSON 외의 어떤 텍스트도 출력 금지, recommend/challenges 중첩/문자열 등 잘못된 구조 방지
- 예시 출력({escaped_format}) 포함, 프롬프트 내 지침 강화

## 2. 라우터 코드(chatbot_router.py) 변경
- SSE 응답에서 LLM 응답 파싱 및 검증 로직 개선
- event: "challenge"/"close"/"error" 등 이벤트별로 JSON 파싱 및 에러 처리 강화
- 최종 응답에서 반드시 challenges가 리스트로 포함되어 있는지 검증
- base-info/free-text 모두 동일한 구조로 챌린지 추천 결과 반환
- 대화 기록/세션 관리 및 카테고리 처리 로직 개선

---