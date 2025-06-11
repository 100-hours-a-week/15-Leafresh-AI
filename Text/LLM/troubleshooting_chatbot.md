# SSE 스트리밍 테스트 트러블슈팅 기록
data[25-06-11]

- 이 문서는 GCP VM 인스턴스에서 SSE(Server-Sent Events) 스트리밍 테스트 중 발생한 문제들과 해결 과정을 기록하였음

# /base-info

## 1. 초기 문제: SSE 연결 타임아웃 (로컬호스트 문제)

### 문제점
`test_sse.html` 파일에서 서버 URL을 `http://localhost:8000`으로 설정했을 때, GCP VM 인스턴스 외부에서 브라우저로 접속 시 타임아웃이 발생했습니다.

### 원인 분석
`netstat -tuln | grep 8000` 명령어를 VM 인스턴스 내부에서 실행했을 때, 서버가 `127.0.0.1:8000`에서만 수신 대기하고 있는 것을 확인했습니다. 이는 서버 애플리케이션이 로컬호스트(VM 내부)에서만 연결을 허용하고, 외부 IP 주소로부터의 트래픽을 수신하지 않기 때문입니다.

### 해결책
*   `test_sse.html`의 URL을 VM의 외부 IP 주소인 `http://35.216.82.57:8000`으로 변경했습니다. (이후 내부 테스트를 위해 다시 `localhost`로 변경)
*   **장기적인 해결책 (외부 접속 시):** 서버를 `uvicorn Text.LLM.main:app --host 0.0.0.0 --port 8000` 명령어로 실행하여 모든 네트워크 인터페이스에서 연결을 수신하도록 설정해야 합니다.

## 2. 모델 경로 오류 (모델 로딩 실패)

### 문제점
서버 시작 시 `uvicorn.log`에서 "모델 경로를 찾을 수 없습니다: /home/ubuntu/15-Leafresh-AI/Text/LLM/Text/LLM/mistral/models--mistralai--Mistral-7B-Instruct-v0.3"와 같은 오류가 발생하며 서버가 충돌했습니다.

### 원인 분석
`Text/LLM/model/chatbot/LLM_chatbot_base_info_model.py` 파일에서 `MODEL_PATH`를 설정할 때, `project_root`가 이미 `/home/ubuntu/15-Leafresh-AI/Text/LLM`으로 설정되어 있는데 `Text`와 `LLM`을 중복하여 추가했기 때문에 잘못된 경로가 생성되었습니다.

### 해결책
`Text/LLM/model/chatbot/LLM_chatbot_base_info_model.py` 파일의 `MODEL_PATH` 설정을 다음과 같이 수정했습니다.
```python
MODEL_PATH = os.path.join(project_root, "mistral", "models--mistralai--Mistral-7B-Instruct-v0.3")
```

## 3. Hugging Face Gated Repo 접근 권한 문제

### 문제점
모델 경로를 수정한 후에도 서버 시작 시 `fastapi.exceptions.HTTPException: 500: 모델 로딩 실패: You are trying to access a gated repo. ... Access to model mistralai/Mistral-7B-Instruct-v0.3 is restricted and you are not in the authorized list.` 오류가 지속되었습니다.

### 원인 분석
`mistralai/Mistral-7B-Instruct-v0.3` 모델은 Hugging Face의 "gated repo"로, 일반적인 공개 모델이 아닙니다. `huggingface_hub.login()`을 통한 API 토큰 인증은 계정 자체의 인증이며, 특정 제한된 모델에 대한 접근 권한을 자동으로 부여하지 않습니다. 이 모델을 사용하려면 Hugging Face 웹사이트에서 별도로 사용 약관에 동의하거나 접근 요청을 제출하여 승인받아야 합니다.

또한, 초기에는 `LLM_chatbot_base_info_model.py` 코드에 `huggingface_hub.login()` 호출 및 `token` 인자 명시적 전달이 누락되어 있었습니다.

### 해결책
1.  **Hugging Face 웹사이트에서 모델 접근 권한 획득:** `https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3` 페이지에 접속하여 "Access gated model" 버튼을 클릭, 약관에 동의하거나 접근 요청을 제출하여 승인을 받았습니다.
2.  **코드에 Hugging Face 로그인 로직 추가:** `Text/LLM/model/chatbot/LLM_chatbot_base_info_model.py` 파일에 다음 코드를 추가하여 `HUGGINGFACE_API_KEYMAC` 환경 변수를 통해 Hugging Face Hub에 로그인하고, `AutoTokenizer.from_pretrained` 및 `AutoModelForCausalLM.from_pretrained` 호출 시 이 토큰을 명시적으로 전달하도록 수정했습니다.

    ```python
    from huggingface_hub import login
    # ...
    load_dotenv()
    # Hugging Face 로그인
    hf_token = os.getenv("HUGGINGFACE_API_KEYMAC")
    if hf_token:
        try:
            login(token=hf_token)
            logger.info("Hugging Face Hub에 성공적으로 로그인했습니다.")
        except Exception as e:
            logger.error(f"Hugging Face Hub 로그인 실패: {e}")
    else:
        logger.warning("HUGGINGFACE_API_KEYMAC 환경 변수를 찾을 수 없습니다. Hugging Face 로그인 건너뜜.")
    # ...
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        cache_dir=MODEL_PATH,
        torch_dtype=torch.float16,
        token=hf_token # 토큰 명시적 전달
    )
    # ...
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        cache_dir=MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        token=hf_token # 토큰 명시적 전달
    )
    ```

## 4. `curl` 명령어 구문 및 인코딩 문제

### 문제점
`curl` 명령어를 사용하여 테스트했을 때, 처음에는 `location은 필수입니다.` 오류가 발생했고, 다음에는 `Invalid HTTP request received.` 오류가 발생했습니다.

### 원인 분석
*   `&` 문자를 작은따옴표로 묶지 않아 쉘에 의해 명령어가 분리되었기 때문입니다.
*   한글 파라미터(`도시`, `사무직`, `비건`)가 URL 인코딩되지 않아 서버가 요청을 올바르게 해석하지 못했습니다.

### 해결책
`curl` 명령어의 URL 전체를 작은따옴표로 묶고, 한글 파라미터를 URL 인코딩하여 다음과 같이 실행하도록 안내했습니다.
```bash
curl -N 'http://localhost:8000/ai/chatbot/recommendation/base-info?sessionId=user123&location=%EB%8F%84%EC%8B%9C&workType=%EC%82%AC%EB%AC%B4%EC%A7%81&category=%EB%B9%84%EA%B1%B4'
```

## 5. LLM 응답 스트리밍 지연 및 로깅 추가

### 문제점
모든 이전 문제가 해결된 후 `curl` 테스트 시 서버 로그에는 `200 OK`가 떴지만, `curl` 터미널에는 즉시 아무런 반응이 없었습니다.

### 원인 분석
LLM 모델이 응답을 생성하는 데 시간이 오래 걸리거나, 스트리밍 이벤트가 제대로 전송되지 않는 문제가 있을 수 있다고 판단했습니다.

### 해결책
`Text/LLM/model/chatbot/LLM_chatbot_base_info_model.py` 파일의 `get_llm_response` 함수 내부에 LLM 추론 과정의 각 단계(입력 준비, 스레드 시작, 토큰 수신, 파싱 시작/완료, 종료 이벤트)에 대한 상세 로그를 추가하여 진행 상황을 추적할 수 있도록 했습니다. 이를 통해 LLM이 실제로 토큰을 생성하고 있음을 확인했습니다.

## 현재 상태
이제 `curl` 명령어를 통해 LLM 응답이 SSE 스트림으로 정상적으로 출력되는 것을 확인했습니다. 내부 서버 테스트는 성공적으로 완료되었습니다.

## 다음 단계 (선택 사항: 외부 접속 테스트)
외부에서 접속하여 테스트하려면:
1.  서버를 `uvicorn Text.LLM.main:app --host 0.0.0.0 --port 8000`으로 재시작하여 모든 IP에서 수신 대기하도록 합니다.
2.  GCP 방화벽에서 8000번 포트에 대한 인그레스 허용 규칙을 추가합니다.
3.  `test_sse.html` 파일의 URL을 VM의 외부 IP 주소(`http://35.216.82.57:8000`)로 변경합니다. 

## 6. LLM 응답 데이터 처리 및 스트리밍 형식 개선

### 6.1. `AttributeError` 및 데이터 구조 문제 해결 (`chatbot_router.py`)

#### 문제점
`chatbot_router.py`의 `event_generator` 함수에서 `challenge` 이벤트를 처리할 때, `data_from_llm_model`이 문자열로 인식되어 `AttributeError: 'str' object has no attribute 'get'` 오류가 발생했습니다. 이는 LLM 모델에서 보낸 토큰 데이터가 예상하는 딕셔너리 형태가 아니었기 때문입니다.

#### 원인 분석
`LLM_chatbot_base_info_model.py`에서 `get_llm_response` 함수는 `format_sse_response("challenge", {"data": {"token": new_text}})` 형태로 데이터를 보냈지만, `chatbot_router.py`에서는 이 `data` 필드를 다시 딕셔너리로 기대했습니다. 또한 `close` 이벤트 시 `data_from_llm_model`이 이미 파싱된 JSON 문자열 또는 객체일 수 있으나, `chatbot_router.py`에서 이를 부적절하게 처리했습니다.

#### 해결책
`Text/LLM/router/chatbot_router.py` 파일의 `event_generator` 함수를 다음과 같이 수정했습니다:
*   `challenge` 이벤트에서는 `data_from_llm_model`이 딕셔너리이고 `token` 키를 포함하는지 확인 후, 토큰과 함께 카테고리/라벨 정보를 클라이언트에 전달하도록 수정했습니다.
*   `close` 이벤트에서는 `data_from_llm_model`을 JSON으로 로드하거나 직접 사용하도록 하여, LLM 모델에서 이미 파싱된 데이터를 재처리하지 않고 직접 활용하도록 했습니다. (참고: 이 시점에는 `data_from_llm_model`이 `LLM_chatbot_base_info_model.py`에서 이미 파싱된 최종 JSON 데이터였습니다.)
*   오류 발생 시 상세 로깅을 추가하여 디버깅을 용이하게 했습니다.

### 6.2. 모델 토큰 생성 `RuntimeError` 해결 (`LLM_chatbot_base_info_model.py`)

#### 문제점
LLM 모델의 토큰 생성 과정에서 `RuntimeError: probability tensor contains invalid values (nan or inf)` 오류가 발생했습니다.

#### 원인 분석
이 오류는 주로 LLM 모델의 불안정한 동작으로 인해 발생하며, 확률 계산에 문제가 생겨 유효하지 않은 값이 생성될 때 나타납니다. 부적절한 세대(generation) 매개변수가 원인일 수 있습니다.

### 6.3. 중첩된 `data` 키로 인한 `ValueError` 해결 (`chatbot_router.py`)

#### 문제점
`파싱된 데이터에 'challenges' 키가 없습니다.`라는 `ValueError`가 발생했습니다.

#### 원인 분석
`LLM_chatbot_base_info_model.py`에서 최종적으로 파싱하여 `close` 이벤트의 `data` 필드로 전달된 JSON 객체가 `{ "status": ..., "message": ..., "data": { "recommend": ..., "challenges": [...] } }`와 같이 실제 챌린지 리스트가 `data`라는 중첩된 키 안에 포함되어 있었습니다. `chatbot_router.py`에서는 이 중첩된 구조를 인지하지 못하고 `parsed_data['challenges']`를 직접 찾으려고 했습니다.

#### 해결책
`Text/LLM/router/chatbot_router.py` 파일의 `event_generator` 함수 내 `close` 이벤트 처리 로직에서 `parsed_data` 내의 `data` 키에 먼저 접근한 후, 그 안에서 `challenges` 키를 찾도록 수정했습니다. (`final_data = parsed_data["data"]` 이후 `final_data["challenges"]` 사용).

### 6.4. SSE `data: data:` 중복 문제 해결 (`chatbot_router.py`)

#### 문제점
`curl`이나 브라우저 개발자 도구에서 SSE 응답을 확인했을 때 `data: data: {...}`와 같이 `data:` 접두사가 중복되어 출력되었습니다.

#### 원인 분석
`FastAPI`의 `sse_starlette.sse.EventSourceResponse`는 `yield`를 통해 `{"event": "이벤트_이름", "data": "데이터_문자열"}` 형태의 딕셔너리를 받으면, 자동으로 SSE 표준 형식(`event: 이벤트_이름
data: 데이터_문자열

`)으로 변환하여 전송합니다. 기존 코드의 `format_sse_response_for_client` 함수가 이미 `event:`와 `data:` 접두사를 포함하는 문자열을 생성하고 있었는데, `EventSourceResponse`가 이 문자열을 다시 `data` 필드로 간주하여 중복 `data:` 접두사를 붙였기 때문입니다.

#### 해결책
`Text/LLM/router/chatbot_router.py` 파일에서 `format_sse_response_for_client` 함수를 제거했습니다. 대신 `event_generator` 함수 내부의 모든 `yield` 문에서 직접 `{"event": "이벤트_이름", "data": json.dumps(데이터_딕셔너리, ensure_ascii=False)}` 형태의 딕셔너리를 반환하도록 수정했습니다. 이로써 `EventSourceResponse`가 올바른 SSE 형식을 생성하게 되어 중복 문제가 해결되었습니다. 

## 7. 카테고리 매핑 및 SSE 응답 형식 개선

### 7.1. 카테고리 매핑 구조 개선

#### 문제점
`base-info` 방식에서 카테고리 정보가 챌린지에 제대로 추가되지 않는 문제가 발생했습니다.

#### 원인 분석
`LLM_chatbot_base_info_model.py`의 `get_llm_response` 함수가 `category` 인자를 받지 않아 `label_mapping`에서 카테고리 정보를 가져올 수 없었습니다.

#### 해결책
1. `Text/LLM/model/chatbot/LLM_chatbot_base_info_model.py` 파일의 `get_llm_response` 함수가 `category` 인자를 받도록 수정했습니다.
2. `Text/LLM/router/chatbot_router.py`에서 `get_base_info_llm_response` 호출 시 `category` 인자를 전달하도록 수정했습니다.
3. `chatbot_constants.py`의 `label_mapping`을 사용하여 각 챌린지에 영어/한글 카테고리 정보를 추가하도록 구현했습니다.

### 7.2. SSE 응답 형식 표준화

#### 문제점
SSE 응답 형식이 API 명세서와 일치하지 않는 문제가 있었습니다.

#### 원인 분석
이전 코드는 `format_sse_response` 함수를 사용하여 SSE 형식을 직접 구성했으나, 이는 FastAPI의 `EventSourceResponse`와 충돌을 일으켰습니다.

#### 해결책
1. `format_sse_response` 함수를 제거하고, 대신 직접 딕셔너리를 yield하도록 변경했습니다.
2. 응답 형식을 API 명세서에 맞게 표준화했습니다:
   ```json
   event: challenge
   data: {
     "status": 200,
     "message": "N번째 챌린지 추천",
     "data": {
       "challenges": {
         "title": "챌린지 제목",
         "description": "챌린지 설명",
         "category": "영어 카테고리",
         "label": "한글 카테고리"
       }
     }
   }
   ```
3. 동기 SSE 방식은 유지하면서, 이벤트 형식만 개선했습니다.

### 7.3. 로깅 개선

#### 문제점
카테고리 매핑 과정에서 발생하는 오류를 추적하기 어려웠습니다.

#### 해결책
1. 파싱 전/후의 JSON 문자열을 로깅하도록 개선했습니다.
2. 카테고리 매핑 과정에서 발생하는 오류에 대한 상세 로깅을 추가했습니다.
3. 전체 응답과 문제가 된 JSON 문자열을 구분하여 로깅하도록 수정했습니다.

## 현재 상태
- 카테고리 매핑이 정상적으로 작동합니다.
- SSE 응답이 API 명세서와 일치하는 형식으로 전송됩니다.
- 로깅이 개선되어 문제 발생 시 디버깅이 용이해졌습니다. 

# /free-text

## 8. free-text 엔드포인트 개선

### 8.1. SSE 응답 형식 표준화

#### 문제점
`free-text` 엔드포인트의 SSE 응답 형식이 `base-info`와 일관되지 않았고, API 명세서와도 일치하지 않았습니다.

#### 원인 분석
- `token` 이벤트와 `complete` 이벤트의 데이터 구조가 불일치했습니다.
- `format_sse_response` 함수를 통해 문자열로 변환된 데이터가 `EventSourceResponse`에 의해 중복 처리되는 문제가 있었습니다.

#### 해결책
1. 이벤트 타입 표준화:
   - `token` 이벤트: 토큰 생성 시점의 데이터
   - `complete` 이벤트: 전체 응답 완료 시점의 데이터
   - `close` 이벤트: 최종 응답 데이터

2. 데이터 구조 표준화:
   ```json
   // token 이벤트
   {
     "status": 200,
     "message": "토큰 생성",
     "data": {
       "token": "생성된 토큰"
     }
   }

   // complete 이벤트
   {
     "status": 200,
     "message": "모든 응답 완료",
     "data": {
       "recommend": "추천 내용",
       "challenges": [...]
     }
   }

   // close 이벤트
   {
     "status": 200,
     "message": "모든 응답 완료",
     "data": {
       "recommend": "추천 내용",
       "challenges": [...]
     }
   }
   ```

3. `format_sse_response` 함수 제거:
   - 직접 딕셔너리를 yield하도록 변경
   - `EventSourceResponse`가 자동으로 SSE 형식으로 변환하도록 수정

### 8.2. 로깅 개선

#### 문제점
`free-text` 엔드포인트에서 LLM 응답 생성 과정을 추적하기 어려웠습니다.

#### 원인 분석
`base-info` 엔드포인트와 달리 `free-text` 엔드포인트에는 상세한 로깅이 없었습니다.

#### 해결책
`Text/LLM/model/chatbot/LLM_chatbot_free_text_model.py` 파일의 `get_llm_response` 함수에 다음과 같은 상세 로깅을 추가했습니다:

1. LLM 응답 생성 시작 시 프롬프트 길이 로깅
2. 토크나이저 입력 준비 완료 시 입력 토큰 수 로깅
3. 모델 생성 스레드 시작 로깅
4. 스트리밍 응답 대기 상태 로깅
5. 각 토큰 수신 시 처음 20자 로깅 (너무 길어지지 않도록 제한)
6. 스트리밍 완료 및 파싱 시작 로깅
7. 파싱 성공 시 응답 데이터 구조 로깅 (recommend 내용 일부와 challenges 개수)

### 현재 상태
- SSE 응답 형식이 `base-info`와 일관되게 표준화되었습니다.
- API 명세서와 일치하는 응답 형식으로 전송됩니다.
- `free-text` 엔드포인트에서도 `base-info`와 동일한 수준의 상세 로깅이 가능합니다.
- LLM 응답 생성 과정의 각 단계를 추적할 수 있어 디버깅이 용이해졌습니다.
- 토큰 생성과 파싱 과정에서 발생하는 문제를 빠르게 파악할 수 있습니다. 