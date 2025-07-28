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

# 2025-06-14 모델 메모리/SSE 속도/양자화 트러블슈팅

## 현상
- FastAPI 기반 LLM 챗봇 서버에서 base-info SSE 응답이 매우 느리고, 메모리 사용량이 90%에 달함.
- 모델: Mistral-7B-Instruct-v0.3 (float16, GPU)
- 서버 실행 시 GPU 메모리 대부분 사용, 응답 속도 저하.

## 실험/분석
- 8bit/4bit 양자화, float16, 오프로드 등 다양한 옵션 실험.
- 8bit/4bit 양자화 시 OOM, 오프로드, 속도 저하 등 반복 발생.
- float16이 오히려 메모리 부족이 덜한 경우도 있었음.
- 양자화 안한 경우보다 8비트 양자화 시 메모리 피크로 실행이 안됐던 오류 발생.
- quantization_config 인자 필수, BitsAndBytesConfig 옵션 다양.
- 코드 구조상 병목 없음, SSE 실시간 구조 최적화 확인.
- 환경(GPU/RAM 공유, 동시 사용, 메모리 단편화, 라이브러리/드라이버 버전, 서버 상태 등)이 속도/성능에 큰 영향.
- 모델 파일(디스크)은 공유 가능, RAM/GPU 메모리는 인스턴스 내 모든 프로세스가 공유.
- 공식 이슈: 8bit가 float16보다 메모리를 더 요구하는 경우도 있음(임시 버퍼 등).

## 결론/권장
- 코드 구조는 실시간 SSE에 최적화되어 있음.
- 속도 저하의 주 원인은 환경(메모리, 오프로드, 동시 사용, 버전, 서버 상태 등).
- 8bit/4bit/float16/더 작은 모델/서버 재부팅/불필요한 프로세스 종료 등 실험 권장.
- "모든 모델이 자동으로 올라가는 것"이 아니라, 코드에서 로딩한 모델만 메모리에 올라감.
- quantization_config 인자는 8bit/4bit 양자화 시 필수. 

# 2025-06-15 모델 생성 설정 및 응답 최적화

## 1. 양자화 설정 변경 (8비트 → 4비트)
- **증상**: 8비트 양자화 시 메모리 부족 및 OOM(Out of Memory) 발생
- **원인**: 
  - 8비트 양자화가 float16보다 메모리를 더 많이 사용하는 경우 발생
  - 임시 버퍼와 추가 메모리 오버헤드로 인한 문제
  - 공식 이슈에서도 보고된 현상
- **해결**: 4비트 양자화로 변경
  ```python
  # 현재 코드 (문제 있음)
  device = "cuda" if torch.cuda.is_available() else "cpu"
  quantization_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_compute_dtype=torch.float16,
      bnb_4bit_use_double_quant=True,
      bnb_4bit_quant_type="nf4"  # fp4에서 nf4로 변경
  )
  ```

## 2. 모델 응답 다양성 문제
- **증상**: 항상 비슷한 형식의 응답만 생성
- **원인**: `do_sample=False`로 설정되어 있어서
- **해결**: 
  - `do_sample=True`로 설정
  - `temperature=0.7`로 조정
  - 다양한 표현의 챌린지 생성 가능

## 3. 메모리 최적화 설정
- **증상**: GPU 메모리 부족 또는 느린 응답
- **해결**:
  ```python
  # 스트리밍 시작 전 메모리 정리
  torch.cuda.empty_cache()
  gc.collect()
  
  # 모델 설정
  model.config.use_cache = False  # 캐시 사용 비활성화
  model.eval()  # 평가 모드로 설정
  ```

## 4. 토큰 생성 설정 설명
```python
generation_kwargs = dict(
    inputs,  # 입력 텐서
    streamer=streamer,  # 스트리밍 응답
    max_new_tokens=1024,  # 최대 생성 토큰 수
    temperature=0.7,  # 생성 다양성 (0.0~1.0)
    do_sample=True,  # 확률적 샘플링
    pad_token_id=tokenizer.eos_token_id  # 패딩 토큰
)
```

## 5. 주의사항
- `max_new_tokens`는 `max_position_embeddings`보다 작아야 함
- `do_sample=False`로 설정하면 `temperature` 값이 무시됨
- 메모리 부족 시 `torch.cuda.empty_cache()`와 `gc.collect()` 사용
- 스트리밍 응답 시 `timeout=None`으로 설정하여 타임아웃 방지

## 현재 상태
- JSON 응답이 정상적으로 완성됨
- 다양한 형식의 챌린지 생성 가능
- 메모리 사용량 최적화
- 스트리밍 응답 안정성 향상

# 2025-06-15 SSE 이벤트 처리 및 중복 요청 트러블슈팅

## 1. 빈 이벤트 전송 방지

### 문제점
- `event:null` 이벤트가 프론트엔드로 전송되는 문제 발생
- 유효하지 않은 이벤트로 인한 프론트엔드 처리 오류

### 원인 분석
- LLM 모델에서 반환되는 이벤트 중 `event_type`이나 `data_from_llm_model`이 없는 경우가 있음
- 이러한 빈 이벤트가 그대로 프론트엔드로 전송되어 문제 발생

### 해결책
`chatbot_router.py`의 이벤트 처리 로직 수정:
```python
if not event_type or not data_from_llm_model:
    continue  # 유효하지 않은 이벤트는 건너뛰기
```

### 현재 상태
- 빈 이벤트가 프론트엔드로 전송되지 않음
- 프론트엔드에서 이벤트 처리가 안정적으로 동작
- 불필요한 네트워크 트래픽 감소

## 2. SSE 이벤트 중복 전송 문제

### 문제점
- `close` 이벤트가 두 번 전송되는 문제 발생
- Spring BE 서버에서 같은 요청이 두 번 들어오는 현상 발견

### 원인 분석
1. FastAPI 서버에서 `close` 이벤트가 두 곳에서 전송됨:
   - 파싱된 데이터와 함께 전송되는 `close` 이벤트
   - `finally` 블록에서 전송되는 추가 `close` 이벤트

2. Spring BE 서버의 SSE 처리:
   ```java
   public SseEmitter stream(String aiUri, Object dto) {
       SseEmitter emitter = new SseEmitter(300_000L);
       sseStreamExecutor.execute(emitter, () ->
           streamHandler.streamToEmitter(emitter, uri)
       );
       return emitter;
   }
   ```
   - `SseEmitter`가 SSE 프로토콜을 자동으로 처리
   - FastAPI의 `EventSourceResponse`가 보내는 SSE 형식을 그대로 전달

### 해결책
2.1. FastAPI 서버 수정:
   ```python
   # finally 블록에서 close 이벤트 제거
   finally:
       # 연결 종료 이벤트는 이미 파싱된 데이터와 함께 전송되었으므로 여기서는 전송하지 않음
       pass
   ```

2.2. 이벤트 타입 표준화:
   - `challenge`: 토큰 생성 시점의 데이터
   - `close`: 파싱된 최종 데이터와 함께 전송
   - `error`: 오류 발생 시 전송

### 현재 상태
- SSE 이벤트가 한 번만 전송됨
- Spring BE 서버에서 SSE 프로토콜이 정상적으로 처리됨
- 클라이언트에서 이벤트를 정상적으로 수신

## 3. 토큰 디코딩 오버플로우 에러

### 문제점
- LLM 모델에서 토큰 생성 중 `OverflowError: out of range integral type conversion attempted` 에러 발생
- 토큰 디코딩 과정에서 정수형 변환 범위 초과 문제

### 원인 분석
```python
# 오류 발생 위치
File "/home/ubuntu/.venv/lib/python3.12/site-packages/transformers/tokenization_utils_fast.py", line 670, in _decode
    text = self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
OverflowError: out of range integral type conversion attempted
```
- 토큰 ID가 Python의 정수형 범위를 초과하는 경우 발생
- 특히 한글과 영어가 혼합된 텍스트에서 자주 발생
- 토큰 캐시 처리 과정에서 메모리 문제 발생 가능

### 해결책
1. **토큰 디코딩 설정 최적화**:
```python
streamer = TextIteratorStreamer(
    tokenizer,
    skip_prompt=True,
    skip_special_tokens=True,
    timeout=None,
    decode_kwargs={
        "skip_special_tokens": True,
        "clean_up_tokenization_spaces": True
    }
)
```

2. **토큰 캐시 관리**:
```python
# 토큰 생성 전 메모리 정리
torch.cuda.empty_cache()
gc.collect()

# 토큰 생성 후 메모리 정리
if hasattr(streamer, 'token_cache'):
    streamer.token_cache = []
```

### 현재 상태
- 토큰 디코딩 오버플로우 에러 발생 빈도 감소
- 메모리 사용량 최적화
- 한글/영어 혼합 텍스트 처리 안정성 향상

## 4. 메모리 관리 최적화

### 문제점
- LLM 모델 운영 중 메모리 누수 및 단편화 발생
- 토큰 생성 과정에서 메모리 사용량 증가
- 여러 요청 처리 시 메모리 부족 현상 발생 가능

### 원인 분석
- 모델 로드 시 이전 메모리가 정리되지 않음
- 토큰 생성 과정에서 캐시가 계속 누적됨
- GPU 메모리 캐시가 적절히 정리되지 않음

### 해결책
4.1. 모델 로드 시점 메모리 정리:
```python
# 메모리 최적화를 위한 설정
torch.cuda.empty_cache()  # GPU 메모리 캐시 비우기
gc.collect()  # 가비지 컬렉션 강제 실행 - 사용하지 않는 메모리 해제 및 메모리 단편화 방지
```

4.2. 답변 생성 완료 후 메모리 정리:
```python
# 토큰 캐시 정리
if hasattr(streamer, 'token_cache'):
    streamer.token_cache = []
```

## 2025-06-16 /tmp 디스크 공간 부족으로 인한 모델 로드 실패

### 문제점
- FastAPI 서버에서 LLM 모델을 로드할 때 `[Errno 28] No space left on device` 에러가 발생하며 서버가 시작되지 않음.
- `/tmp` 디스크가 100% 사용 중이었음.

### 원인 분석
- `/tmp`는 리눅스 시스템에서 임시 파일, 캐시, 압축 해제 파일, 소켓 등 다양한 임시 데이터를 저장하는 공간임.
- Hugging Face transformers, PyTorch 등에서 대용량 모델을 로드할 때 모델 파일을 임시로 압축 해제하거나, 캐시 파일을 만들거나, 메모리가 부족할 때 일부 데이터를 임시로 저장할 수 있음.
- 서버가 오래 켜져 있거나, 여러 번 모델을 로드/언로드 하다 보면 `/tmp`에 임시 파일이 쌓여 공간이 부족해질 수 있음.
- `/tmp`가 가득 차 있으면 모델 로드/추론/캐시 생성이 실패할 수 있음.

### 해결 과정
1. `df -h` 명령어로 디스크 사용량을 확인하여 `/tmp`가 100% 사용 중임을 확인.
2. `rm -rf ~/.cache/* && rm -rf /tmp/ubuntu/*` 명령어로 불필요한 캐시 및 임시 파일을 정리.
3. 정리 후 `/tmp` 사용량이 1%로 줄고, 전체 디스크 사용량도 감소함을 확인.
4. 이후 FastAPI 서버를 재시작하니 모델이 정상적으로 로드됨.

### 실무 팁
- 대용량 모델을 자주 다루는 서버라면 `/tmp`의 용량을 넉넉하게 잡거나, 별도의 임시 디렉토리를 지정하는 것이 좋음.
- 모델 로드 시 `cache_dir`나 `offload_folder`를 `/tmp`가 아닌, 용량이 넉넉한 디렉토리로 지정할 수도 있음.
- 주기적으로 `/tmp`를 정리해주는 것이 좋음.

## 2025-06-17 SSE 연결 종료 처리 개선

### 문제점
- LLM 모델 로딩 및 응답 생성 과정에서 불필요하게 중복된 메모리 정리 로직이 존재
- `torch.cuda.empty_cache()`와 `gc.collect()`가 여러 곳에서 호출되어 오히려 성능에 영향을 줄 수 있음

### 원인 분석
- 모델 로드 시점에만 한 번 메모리 정리를 수행하면 충분함
- 스트리밍 응답 처리 중에는 모델이 이미 로드되어 있으므로 추가적인 메모리 정리가 대부분 불필요함

### 해결 과정
`Text/LLM/model/chatbot/LLM_chatbot_base_info_model.py` 파일의 모델 로드 부분에서 중복되거나 불필요한 메모리 정리 로직을 제거하고, 모델 로드 전에만 한 번 실행되도록 통합:

```python
# 모델 로드 전 메모리 정리
torch.cuda.empty_cache()
gc.collect()

# 모델 로드 시 메모리 최적화 옵션 추가
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    cache_dir=MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    token=hf_token,
    trust_remote_code=True,
    max_position_embeddings=2048,
    quantization_config=quantization_config,
    offload_folder="offload",
    offload_state_dict=True
)

# 메모리 최적화를 위한 설정 (유지)
model.config.use_cache = False
model.eval()

# ... 기존 코드 (get_llm_response 함수 내부에서 불필요한 메모리 정리 제거) ...

def get_llm_response(prompt: str, category: str) -> Generator[Dict[str, Any], None, None]:
    # ... 기존 코드 ...
    # 스트리밍 시작 전 메모리 정리 (이 부분 제거)
    # torch.cuda.empty_cache()
    # gc.collect()

    # ... 기존 코드 ...

    # 스레드 완료 대기
    thread.join()

    # 토큰 캐시 정리 (필요시 유지)
    if hasattr(streamer, 'token_cache'):
        streamer.token_cache = []

    # 메모리 정리 (이 부분 제거)
    # torch.cuda.empty_cache()
    # gc.collect()

    # ... 나머지 코드 ...
```

### 실무 팁
- `torch.cuda.empty_cache()`와 `gc.collect()`는 필요한 시점에 한 번만 호출하는 것이 효율적임.
- 모델 로드 시점에 메모리를 최적화하고, 이후 불필요한 호출을 자제하여 성능 오버헤드를 줄임.
- `model.config.use_cache = False`와 같은 모델 자체의 캐시 설정을 활용하여 메모리 사용량을 관리.

### 문제점
- /base-info의 FastAPI 서버에서 SSE 스트리밍 응답 처리 중 `RuntimeError: Unexpected ASGI message 'http.response.body' sent, after response already completed` 에러 발생
- 모델의 응답은 정상적으로 생성되고 파싱되었으나, 클라이언트로 전송하는 과정에서 문제 발생

### 원인 분석
- SSE 스트리밍이 완전히 종료되기 전에 연결이 끊어짐
- 응답이 이미 완료된 상태에서 추가 응답을 보내려고 시도
- 이는 주로 다음과 같은 상황에서 발생:
  1. 클라이언트가 연결을 일찍 종료
  2. 네트워크 연결 불안정
  3. 서버의 응답 처리 로직이 비정상적으로 종료

### 해결 과정
1. `response_completed` 플래그 추가: 응답이 완료되었는지 추적
2. 스트리밍 처리 시 `response_completed` 체크: 응답이 완료되지 않은 경우에만 처리
3. 응답 완료 시점에 `response_completed = True` 설정
4. 에러 발생 시에도 `response_completed = True` 설정

```python
full_response = ""
logger.info("스트리밍 응답 대기 중...")
response_completed = False  # 응답 완료 여부를 추적하는 플래그

try:
    # 스트리밍 응답 처리
    for new_text in streamer:
        if new_text and not response_completed:  # 응답이 완료되지 않은 경우에만 처리
            full_response += new_text
            logger.info(f"토큰 수신: {new_text[:20]}...")
            
            # ... 토큰 처리 로직 ...

            if not response_completed:
                response_completed = True
                yield {
                    "event": "close",
                    "data": json.dumps({
                        "status": 200,
                        "message": "모든 챌린지 추천 완료",
                        "data": parsed_data
                    }, ensure_ascii=False)
                }
```

## 코드 최적화 및 안정성 개선

### 1. 양자화 설정 변경 (fp4 → nf4)
- **증상**: fp4 양자화 타입이 CPU에서 지원되지 않는 문제 발생
- **원인**: fp4는 GPU 전용 양자화 타입으로, CPU 환경에서 오류 발생
- **해결**: nf4(Normal Float 4) 양자화 타입으로 변경
  ```python
  quantization_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_compute_dtype=torch.float16,
      bnb_4bit_use_double_quant=True,
      bnb_4bit_quant_type="nf4"  # fp4에서 nf4로 변경
  )
  ```

### 2. 메모리 관리 개선
- **증상**: 메모리 누수 및 불필요한 메모리 정리로 인한 성능 저하
- **해결**: finally 블록 추가로 메모리 정리 로직 통합
  ```python
  finally:
      # 요청 완료 후 메모리 정리
      try:
          if 'inputs' in locals():
              del inputs
          torch.cuda.empty_cache()
          gc.collect()
          logger.info("메모리 정리 완료")
      except Exception as e:
          logger.error(f"메모리 정리 중 에러 발생: {str(e)}")
  ```

### 3. /free-text 엔드포인트 비동기 전환
- **증상**: 동기 처리로 인한 응답 지연
- **해결**: async/await 패턴으로 비동기 처리 전환
  ```python
  @router.get("/ai/chatbot/recommendation/free-text")
  async def freetext_rag(
      sessionId: Optional[str] = Query(None),
      message: Optional[str] = Query(None)
  ):
      # 비동기 처리 로직 구현
  ```

### 4. 파싱 로직 개선 및 스트리밍 안정화
- **증상**: JSON 파싱 오류 및 스트리밍 데이터 처리 문제, inf/nan 값 발생, event: token → challenge로 이벤트 타입 변경 필요
- **해결**:
  1. 스트리머 설정 최적화
  2. inf/nan 값 처리를 위한 로짓 프로세서 추가
  3. event: token → challenge로 이벤트 타입 변경
  ```python
  # 로짓 프로세서 설정
  logits_processor = LogitsProcessorList([
      InfNanRemoveLogitsProcessor()
  ])
  
  # 스트리머 설정
  streamer = TextIteratorStreamer(
      tokenizer,
      skip_prompt=True,
      skip_special_tokens=True,
      timeout=None,
      decode_kwargs={
          "skip_special_tokens": True,
          "clean_up_tokenization_spaces": True,
          "errors": "ignore"
      }
  )
  ```

### 현재 상태
- 양자화 설정이 CPU/GPU 환경 모두에서 안정적으로 동작
- 메모리 관리가 효율적으로 이루어짐
- /free-text 엔드포인트의 응답 속도 개선
- JSON 파싱 및 스트리밍 데이터 처리 안정성 향상
- inf/nan 값으로 인한 오류 발생 빈도 감소

# 2025-06-18 
# 공유 모델 구현 및 메모리 최적화

## 1. 메모리 중복 사용 문제 해결

### 문제점
- 두 엔드포인트(`/base-info`, `/free-text`)가 각각 독립적으로 모델을 로드
- 각각 4GB씩 사용하여 총 8GB 메모리 사용 (L4 GPU 24GB 중 33% 사용)
- 연속 요청 시 메모리 부족으로 CUDA Out of Memory 오류 발생
- 첫 번째 요청 후 메모리가 제대로 정리되지 않아 두 번째 요청 시 문제 발생

### 원인 분석
```python
# 기존 구조 (문제 있음)
# LLM_chatbot_base_info_model.py: 모델 로드 (4GB)
# LLM_chatbot_free_text_model.py: 또 다른 모델 로드 (4GB)
# 총 8GB 사용
```

### 해결책
**싱글톤 패턴의 공유 모델 구현**:
1. `Text/LLM/model/chatbot/shared_model.py` 생성
2. `SharedMistralModel` 클래스로 싱글톤 패턴 구현
3. 두 엔드포인트가 같은 모델 인스턴스를 공유

```python
class SharedMistralModel:
    _instance = None
    _model = None
    _tokenizer = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SharedMistralModel, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialize_model()
            self._initialized = True
```

## 2. 8비트 vs 4비트 양자화 실험

### 8비트 양자화 시도
```python
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=True,
    llm_int8_enable_fp32_cpu_offload=True
)
```

### 문제점
- GPU 메모리 부족으로 CPU/디스크 오프로드 필요
- 오류: "Some modules are dispatched on the CPU or the disk"
- L4 GPU에서 8GB 모델 + 임시 버퍼로 인한 메모리 부족

### 최종 결정: 4비트 양자화 유지
```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
```

## 3. 코드 리팩토링

### 3.1. 중복 코드 제거
- `LLM_chatbot_base_info_model.py`에서 모델 로딩 코드 제거
- `LLM_chatbot_free_text_model.py`에서 모델 로딩 코드 제거
- 공유 모델 사용으로 변경

```python
# 변경 전
# 각 파일에서 독립적으로 모델 로드

# 변경 후
from Text.LLM.model.chatbot.shared_model import shared_model
model = shared_model.model
tokenizer = shared_model.tokenizer
```

### 3.2. 메모리 정리 통합
```python
def cleanup_memory(self):
    """메모리 정리"""
    try:
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Shared model memory cleanup completed")
    except Exception as e:
        logger.error(f"Memory cleanup error: {str(e)}")
```

## 4. 성능 개선 효과

### 메모리 사용량
- **기존**: 8GB (두 개 모델)
- **개선 후**: 4GB (하나의 공유 모델)
- **절약**: 50% 메모리 절약

### 안정성
- 연속 요청 시 메모리 부족 오류 해결
- CUDA 메모리 단편화 문제 감소
- 서버 재시작 없이도 안정적인 운영 가능

### 유지보수성
- 모델 설정 변경 시 한 곳에서만 수정
- 코드 중복 제거
- 일관된 메모리 관리

# 토큰 제한 및 대화 기록 최적화

## 1. 입력 토큰 제한 초과 문제

### 문제점
- 사용자 입력이 모델의 최대 토큰 제한(2048)을 초과하는 경우 발생
- 오류: "입력이 너무 깁니다. 최대 2048 토큰까지 허용됩니다. 현재: XXXX 토큰"
- 대화 기록이 누적되면서 토큰 수가 급격히 증가

### 원인 분석
- Mistral-7B 모델의 `max_position_embeddings=2048` 설정
- 대화 기록이 계속 누적되어 토큰 수 증가
- 프롬프트 템플릿에 컨텍스트, 쿼리, 메시지 히스토리가 모두 포함

### 해결책
1. **동적 토큰 수 체크 및 대화 기록 조정**:
```python
# 토큰 수 체크 및 대화 기록 조정
messages = current_state["messages"]
while len(messages) > 2:  # 최소 1번의 대화는 유지
    # 현재 메시지들로 프롬프트 구성
    test_messages = "\n".join(messages)
    test_prompt = custom_prompt.format(
        context="",
        query=query,
        messages=test_messages,
        category=current_state["category"]
    )
    
    # 토큰 수 체크
    test_inputs = tokenizer(test_prompt, return_tensors="pt")
    if test_inputs.input_ids.shape[1] <= 1800:  # 여유를 두고 1800 토큰으로 제한
        break
    
    # 토큰 수가 많으면 가장 오래된 대화 제거 (2개씩: User + Assistant)
    messages = messages[2:]
```

2. **대화 기록 제한 강화**:
```python
# 대화 기록이 너무 길어지면 오래된 메시지 제거 (더 엄격하게 제한)
if len(current_state["messages"]) > 6:  # 10개에서 6개로 줄임 (3번의 대화)
    current_state["messages"] = current_state["messages"][-6:]
```

## 2. 프론트엔드 요청 인코딩 문제

### 문제점
- 프론트엔드에서 "아무거나 추천" 같은 요청이 fallback 메시지로 처리됨
- 예상: LLM 응답
- 실제: "저는 친환경 챌린지를 추천해드리는 Leafresh 챗봇이에요!..."

### 원인 분석
- 프론트엔드 요청이 이중 URL 인코딩되어 전송됨
- `ENV_KEYWORDS` 체크에서 "아무거나"가 인식되지 않음
- fallback 로직이 잘못 트리거됨

### 해결책
1. **URL 디코딩 추가**(/free-text):
```python
from urllib.parse import unquote  # URL 디코딩을 위한 import 추가

# URL 디코딩 추가
if message:
    message = unquote(message)
```

2. **환경 키워드 확장**:
```python
ENV_KEYWORDS = [
    # 기존 키워드들...
    "아무거나", "무엇", "뭐", "추천", "추천해", "추천해줘",
    "환경", "친환경", "지구", "탄소", "배출", "절약", "재활용",
    "플라스틱", "일회용", "에너지", "전기", "물", "음식물",
    "교통", "대중교통", "자전거", "도보", "카풀", "전기차",
    "비건", "채식", "로컬", "유기농", "제로웨이스트", "미니멀",
    "업사이클", "리사이클", "컴포스트", "텀블러", "장바구니",
    "친환경", "지속가능", "탄소중립", "기후변화", "오염",
    "자연", "생태", "보호", "보존", "청결", "깨끗"
]
```

## 3. SSE 이벤트 처리 개선

### 문제점
- 스트리밍 응답 중 불필요한 토큰 정제로 인한 데이터 손실
- JSON 구조 제거 과정에서 실제 내용까지 제거되는 문제

### 해결책
**토큰 정제 로직 개선**:
```python
# 토큰 정제 - 순수 텍스트만 추출
cleaned_text = new_text
# JSON 관련 문자열 제거
cleaned_text = re.sub(r'"(recommend|challenges|title|description)":\s*("|\')?', '', cleaned_text)
# 마크다운 및 JSON 구조 제거
cleaned_text = cleaned_text.replace("```json", "").replace("```", "").strip()
cleaned_text = re.sub(r'["\']', '', cleaned_text)  # 따옴표 제거
cleaned_text = re.sub(r'[\[\]{}]', '', cleaned_text)  # 괄호 제거
cleaned_text = re.sub(r',\s*$', '', cleaned_text)  # 끝의 쉼표 제거
# 불필요한 공백 제거
cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
cleaned_text = cleaned_text.strip()
```

## 4. 로깅 및 디버깅 개선

### 추가된 로깅
1. **토큰 수 체크 로깅**:
```python
logger.info(f"토크나이저 입력 준비 완료. 입력 토큰 수: {inputs.input_ids.shape[1]}")
```

2. **대화 기록 조정 로깅**:
```python
logger.info(f"대화 기록 조정: {len(messages)}개 메시지 유지")
```

3. **URL 디코딩 로깅**:
```python
logger.info(f"원본 메시지: {message}")
logger.info(f"디코딩된 메시지: {unquote(message)}")
```

## 5. 성능 최적화 효과

### 토큰 관리
- 동적 토큰 수 체크로 메모리 효율성 향상
- 대화 기록 제한으로 응답 속도 개선
- 안전한 토큰 제한(1800)으로 오류 방지

### 사용자 경험
- URL 인코딩 문제 해결로 자연스러운 대화 가능
- 환경 키워드 확장으로 더 많은 쿼리 인식
- fallback 로직 정확성 향상

### 안정성
- 토큰 제한 초과 오류 방지
- 메모리 사용량 예측 가능
- 일관된 응답 품질 유지

## 현재 상태
- 토큰 제한 문제 완전 해결
- 프론트엔드 요청 정상 처리
- 대화 기록 최적화로 성능 향상
- 로깅 개선으로 디버깅 용이성 증대

# challenges 파싱 오류 해결

## 1. challenges 문자열 파싱 문제

### 문제점
- LLM 응답에서 `challenges` 필드가 문자열로 파싱되는 경우 발생
- 오류: `'str' object does not support item assignment`
- JSON 파싱은 성공했지만 challenges를 리스트로 변환하지 않고 딕셔너리로 접근 시도

### 원인 분석
```python
# 로그에서 확인된 문제
"challenges": "[\n    {\n        \"title\": \"전기차 소비 챌린지\",\n        \"description\": \"...\"\n    }\n]"
```
- `challenges` 필드가 JSON 문자열로 파싱됨
- 이를 리스트로 변환하지 않고 바로 딕셔너리로 접근하려고 시도
- `for challenge in parsed_data["challenges"]:`에서 오류 발생

### 해결책
1. **challenges 문자열 감지 및 변환**:
```python
# challenges가 문자열인 경우 리스트로 변환
if isinstance(parsed_data["challenges"], str):
    challenges_list = parse_challenges_string(parsed_data["challenges"])
    parsed_data["challenges"] = challenges_list
    logger.info(f"challenges 문자열을 리스트로 변환 완료: {len(challenges_list)}개 챌린지")
```

2. **타입 검증 강화**:
```python
# challenges가 리스트인지 확인
if isinstance(parsed_data["challenges"], list):
    for challenge in parsed_data["challenges"]:
        challenge["category"] = eng_label
```

3. **parse_challenges_string 함수 활용**:
```python
def parse_challenges_string(challenges_str: str) -> list:
    """challenges 문자열을 파싱하여 리스트로 변환"""
    # 이미 리스트인 경우 그대로 반환
    if isinstance(challenges_str, list):
        return challenges_str
    
    # JSON 파싱 시도
    try:
        return json.loads(challenges_str)
    except:
        pass
    
    # 문자열 파싱 로직...
    return challenges
```

## 2. JSON 파싱 최적화

### 문제점
- `json.loads(response)`를 여러 번 호출하면서 딕셔너리를 수정하려고 시도
- 불필요한 JSON 파싱으로 인한 성능 저하

### 해결책
```python
# JSON 응답을 한 번만 파싱
response_data = json.loads(response)

# 필수 필드 검증
if "recommend" not in response_data or "challenges" not in response_data:
    raise ValueError("응답에 필수 필드가 없습니다.")

# challenges가 문자열인 경우 배열로 변환
if isinstance(response_data.get("challenges"), str):
    challenges = parse_challenges_string(response_data["challenges"])
    response_data["challenges"] = challenges
```

## 3. 성능 최적화 효과

### 안전성
- 타입 검증으로 런타임 오류 방지
- 다양한 형태의 challenges 응답 처리 가능
- 기존 `parse_challenges_string` 함수 재활용으로 코드 일관성 유지

### 디버깅
- 상세한 로깅으로 디버깅 용이성 향상
- challenges 변환 과정 추적 가능
- 오류 발생 시 명확한 원인 파악 가능

# 2025-06-18 토큰 디코딩 오버플로우 에러 강화 처리

## 1. OverflowError 재발생 문제

### 문제점
- 토큰 디코딩 과정에서 `OverflowError: out of range integral type conversion attempted` 오류 재발생
- 스레드가 완전히 중단되어 응답 처리가 불가능한 상황
- 한글과 영어가 혼합된 텍스트에서 자주 발생하는 문제

### 원인 분석
```python
# 오류 발생 위치
File "/home/ubuntu/.venv/lib/python3.12/site-packages/transformers/tokenization_utils_fast.py", line 670, in _decode
    text = self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
OverflowError: out of range integral type conversion attempted
```
- 토큰 ID가 Python의 정수형 범위를 초과하는 경우 발생
- 특히 한글과 영어가 혼합된 텍스트에서 자주 발생
- 토큰 캐시 처리 과정에서 메모리 문제 발생 가능

### 해결책
1. **토큰 디코딩 설정 최적화**:
```python
streamer = TextIteratorStreamer(
    tokenizer,
    skip_prompt=True,
    skip_special_tokens=True,
    timeout=None,
    decode_kwargs={
        "skip_special_tokens": True,
        "clean_up_tokenization_spaces": True
    }
)
```

2. **토큰 캐시 관리**:
```python
# 토큰 생성 전 메모리 정리
torch.cuda.empty_cache()
gc.collect()

# 토큰 생성 후 메모리 정리
if hasattr(streamer, 'token_cache'):
    streamer.token_cache = []
```

### 현재 상태
- 토큰 디코딩 오버플로우 에러 발생 빈도 감소
- 메모리 사용량 최적화
- 한글/영어 혼합 텍스트 처리 안정성 향상

## 2. 스트리밍 안정성 개선

### 문제점
- 오버플로우 에러 발생 시 스레드가 완전히 중단됨
- 현재까지 수집된 응답도 손실되는 문제
- 사용자에게 빈 응답 또는 오류만 전달됨

### 해결책
1. **부분 응답 처리**:
- 오버플로우 에러 발생 시 현재까지 수집된 응답을 활용
- 완전한 JSON이 아니어도 가능한 부분까지 파싱 시도
- 사용자에게 의미 있는 응답 제공

2. **에러 복구 메커니즘**:
```python
# 스레드 완료 대기
thread.join()

# 토큰 캐시 정리
if hasattr(streamer, 'token_cache'):
    streamer.token_cache = []

# 전체 응답 파싱 (오버플로우 에러가 발생해도 현재까지의 응답 처리)
if full_response and not response_completed:
    # ... 응답 처리 로직 ...
```

## 3. 로깅 개선

### 추가된 로깅
1. **오버플로우 에러 로깅**:
```python
logger.error(f"토큰 디코딩 오버플로우 에러 발생: {str(e)}")
logger.info("오버플로우 에러로 인해 스트리밍을 중단하고 현재까지의 응답을 처리합니다.")
```

2. **응답 파싱 에러 로깅**:
```python
logger.error(f"응답 파싱 중 에러 발생: {str(e)}")
```

## 4. 성능 최적화 효과

### 안정성
- 오버플로우 에러 발생 시에도 부분 응답 처리 가능
- 스레드 중단 없이 안정적인 응답 생성
- 사용자 경험 개선

### 복구 능력
- 토큰 디코딩 실패 시에도 현재까지의 응답 활용
- 완전한 JSON이 아니어도 가능한 부분까지 파싱
- 에러 발생 시에도 의미 있는 응답 제공

### 메모리 관리
- 토큰 캐시 정리로 메모리 누수 방지
- 스레드 완료 대기로 리소스 정리 보장
- 안정적인 메모리 사용

## 현재 상태
- 오버플로우 에러 완전 처리
- 부분 응답 처리로 안정성 향상
- 사용자 경험 개선
- 메모리 관리 최적화

# feedback 모델 공유 모델 적용

## 1. Feedback 모델 메모리 중복 사용 문제

### 문제점
- Feedback 모델이 독립적으로 모델을 로드하여 메모리 중복 사용
- 기존: Chatbot 모델들 (8GB) + Feedback 모델 (4GB) = 총 12GB 메모리 사용
- L4 GPU 24GB 중 50% 사용으로 메모리 부족 위험 증가
- 연속 요청 시 메모리 부족으로 CUDA Out of Memory 오류 발생 가능

### 원인 분석
```python
# 기존 구조 (문제 있음)
# LLM_chatbot_base_info_model.py: 모델 로드 (4GB)
# LLM_chatbot_free_text_model.py: 모델 로드 (4GB)  
# LLM_feedback_model.py: 또 다른 모델 로드 (4GB)
# 총 12GB 사용
```

### 해결책
**Feedback 모델도 공유 모델 사용으로 변경**:
1. `Text/LLM/model/feedback/LLM_feedback_model.py` 수정
2. 독립적인 모델 로딩 코드 제거
3. 공유 모델의 model과 tokenizer 사용

```python
# 변경 전 (독립적 모델 로딩)
class FeedbackModel:
    def __init__(self):
        # Hugging Face 로그인
        hf_token = os.getenv("HUGGINGFACE_API_KEYMAC")
        if hf_token:
            login(token=hf_token)
        
        # 토크나이저 로딩
        self.tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.3",
            cache_dir=MODEL_PATH,
            torch_dtype=torch.float16,
            token=hf_token
        )
        
        # 모델 로딩
        self.model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.3",
            cache_dir=MODEL_PATH,
            device_map="auto",
            low_cpu_mem_usage=True,
            token=hf_token,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            max_position_embeddings=2048,
            quantization_config=quantization_config,
            offload_folder="offload",
            offload_state_dict=True
        )

# 변경 후 (공유 모델 사용)
from Text.LLM.model.chatbot.shared_model import shared_model

class FeedbackModel:
    def __init__(self):
        # 공유 모델 사용으로 변경
        self.model = model
        self.tokenizer = tokenizer
        
        logger.info("Feedback model initialized with shared model")
```

## 2. 메모리 관리 통합

### 문제점
- Feedback 모델에서 독립적인 메모리 정리 로직 사용
- `torch.cuda.empty_cache()`와 `gc.collect()` 중복 호출
- 일관성 없는 메모리 관리

### 해결책
**공유 모델의 메모리 정리 함수 사용**:
```python
async def generate_feedback(self, data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        # 메모리 정리
        shared_model.cleanup_memory()
        
        # ... 피드백 생성 로직 ...
        
    except Exception as e:
        # ... 에러 처리 ...
    finally:
        # 메모리 정리
        shared_model.cleanup_memory()
        logger.info("Feedback model memory cleanup completed")
```

## 3. 코드 리팩토링

### 3.1. 중복 코드 제거
- Hugging Face 로그인 코드 제거 (공유 모델에서 이미 처리)
- 토크나이저 로딩 코드 제거
- 모델 로딩 코드 제거
- 양자화 설정 코드 제거
- GPU 메모리 계산 코드 제거

### 3.2. 의존성 정리
```python
# 제거된 import들
# from huggingface_hub import login
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 유지된 import들
from Text.LLM.model.chatbot.shared_model import shared_model
```

## 4. 성능 개선 효과

### 메모리 사용량
- **기존**: 12GB (3개 모델)
- **개선 후**: 4GB (1개 공유 모델)
- **절약**: 67% 메모리 절약

### 안정성
- 연속 요청 시 메모리 부족 오류 해결
- CUDA 메모리 단편화 문제 감소
- 서버 재시작 없이도 안정적인 운영 가능

### 유지보수성
- 모델 설정 변경 시 한 곳에서만 수정
- 코드 중복 제거
- 일관된 메모리 관리

### 응답 속도
- 모델 로딩 시간 단축 (이미 로드된 모델 사용)
- 메모리 정리 오버헤드 감소
- 전반적인 성능 향상

## 5. 메모리 정리 위치 최적화

### 함수 시작 시 메모리 정리
```python
async def generate_feedback(self, data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        # 메모리 정리
        shared_model.cleanup_memory()
        
        # 입력 데이터 검증...
```
**이유:**
- 이전 요청의 메모리 잔여물 정리
- 깨끗한 상태에서 피드백 생성 시작
- 메모리 단편화 방지

### 함수 종료 시 메모리 정리
```python
    finally:
        # 메모리 정리
        shared_model.cleanup_memory()
        logger.info("Feedback model memory cleanup completed")
```
**이유:**
- 성공/실패 관계없이 확실한 메모리 정리 보장
- 다음 요청을 위한 준비
- 메모리 누수 방지

## 현재 상태
- Feedback 모델 공유 모델 적용 완료
- 메모리 사용량 67% 절약 (12GB → 4GB)
- 일관된 메모리 관리로 안정성 향상
- 코드 중복 제거로 유지보수성 개선

# 2025-07-03 vLLM 도입 및 코드 구조 변경

## 1. vLLM 설치 및 설정

### vLLM 도입 배경
- 기존: Hugging Face Transformers를 직접 사용하여 모델 로딩 및 추론
- 문제점: 메모리 사용량 대비 추론 속도가 느림, 동시 요청 처리 시 병목 현상
- 해결책: vLLM(Very Large Language Model) 도입으로 추론 성능 최적화

### vLLM 설치 과정
```bash
# vLLM 설치
pip install vllm

# vLLM 서버 시작 (별도 프로세스로 실행)
python -m vllm.entrypoints.openai.api_server \
    --model /home/ubuntu/mistral/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/e0bc86c23ce5aae1db576c8cca6f06f1f73af2db \
    --port 8800 \
    --host 0.0.0.0
```

### vLLM 서버 설정
- **포트**: 8800번 포트에서 OpenAI API 호환 인터페이스 제공
- **모델 경로**: 기존 Hugging Face 모델 경로 그대로 사용
- **호스트**: 0.0.0.0으로 설정하여 외부 접근 허용

## 2. 코드 구조 변경사항

### 2.1. 모델 로딩 방식 변경

#### 기존 코드 (Hugging Face Transformers 직접 사용)
```python
# 공유 모델 사용
model = shared_model.model
tokenizer = shared_model.tokenizer

def get_llm_response(prompt: str, category: str) -> Generator[Dict[str, Any], None, None]:
    # 메모리 정리
    shared_model.cleanup_memory()
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 스트리머 설정
    streamer = TextIteratorStreamer(tokenizer, ...)
    
    # 모델 생성 설정
    generation_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=1024,
        temperature=0.7,
        do_sample=True,
        # ... 기타 설정
    )
    
    # 스레드로 모델 생성
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # 스트리밍 응답 처리
    for new_text in streamer:
        # ... 응답 처리
```

#### 새로운 코드 (vLLM HTTP API 사용)
```python
# vLLM 서버 호출용 httpx 사용
import httpx

def get_llm_response(prompt: str, category: str) -> Generator[Dict[str, Any], None, None]:
    """vLLM 서버에 POST 요청하여 응답을 SSE 형식으로 반환"""
    logger.info(f"[vLLM 호출] 프롬프트 길이: {len(prompt)}")
    url = "http://localhost:8800/v1/chat/completions"
    payload = {
        "model": "/home/ubuntu/mistral/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/e0bc86c23ce5aae1db576c8cca6f06f1f73af2db",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True
    }

    response_completed = False  # 응답 완료 여부를 추적하는 플래그

    try:
        with httpx.stream("POST", url, json=payload, timeout=60.0) as response:
            full_response = ""
            for line in response.iter_lines():
                if line.startswith(b"data: "):
                    try:
                        json_data = json.loads(line[len(b"data: "):])
                        delta = json_data["choices"][0]["delta"]
                        token = delta.get("content", "")
                        # ... 토큰 처리
```

### 2.2. 주요 변경사항

#### 2.2.1. 모델 로딩 제거
- **기존**: `shared_model.model`, `shared_model.tokenizer` 직접 사용
- **변경**: vLLM 서버가 모델을 관리하므로 로컬 모델 로딩 불필요
- **효과**: 메모리 사용량 대폭 감소, 서버 시작 시간 단축

#### 2.2.2. HTTP API 통신 방식
- **기존**: Python 스레드와 TextIteratorStreamer를 사용한 직접 추론
- **변경**: httpx를 사용한 HTTP 스트리밍 통신
- **효과**: 더 안정적인 스트리밍, 동시 요청 처리 개선

#### 2.2.3. 토큰 처리 방식
- **기존**: TextIteratorStreamer에서 직접 토큰 디코딩
- **변경**: vLLM 서버에서 JSON 형태로 토큰 전송
- **효과**: 토큰 디코딩 오버플로우 에러 해결

#### 2.2.4. 메모리 관리 단순화
- **기존**: 복잡한 메모리 정리 로직 (torch.cuda.empty_cache(), gc.collect())
- **변경**: vLLM 서버가 메모리 관리하므로 로컬 메모리 정리 불필요
- **효과**: 코드 단순화, 메모리 관리 오버헤드 제거

### 2.3. JSON 파싱 로직 대폭 개선

#### 2.3.1. 중복 JSON 제거 로직 추가
**문제점**: LLM이 여러 개의 JSON을 한 번에 생성하여 파싱 오류 발생
**해결책**: 첫 번째 완전한 JSON만 추출하는 로직 구현

#### 기존 코드
```python
# JSON 문자열 추출
json_match = re.search(r"```json\n([\s\S]*?)\n```", full_response.strip())
if json_match:
    json_string_to_parse = json_match.group(1).strip()
else:
    json_string_to_parse = full_response.strip()

# 복잡한 정제 과정
json_string_to_parse = re.sub(r',(\s*[}\]])', r'\1', json_string_to_parse)
json_string_to_parse = re.sub(r',\s*,', ',', json_string_to_parse)
# ... 기타 정제 과정
```

#### 새로운 코드
```python
# 중복 JSON 제거 - 첫 번째 완전한 JSON만 추출
json_objects = []
brace_count = 0
start_idx = -1

for i, char in enumerate(json_str):
    if char == '{':
        if brace_count == 0:
            start_idx = i
        brace_count += 1
    elif char == '}':
        brace_count -= 1
        if brace_count == 0 and start_idx != -1:
            json_obj = json_str[start_idx:i+1]
            try:
                json.loads(json_obj)
                json_objects.append(json_obj)
                break
            except:
                continue

if json_objects:
    json_str = json_objects[0]
else:
    # 기존 방식으로 fallback
    if "{" in json_str and "}" in json_str:
        json_str = json_str[json_str.find("{"):json_str.rfind("}")+1]

# 추가 정제
json_str = re.sub(r',\s*([}\]])', r'\1', json_str)  # 마지막 쉼표 제거
json_str = re.sub(r',\s*,', ',', json_str)          # 연속 쉼표 제거
json_str = re.sub(r'[ \t\r\f\v]+', ' ', json_str)   # 공백 정규화
```

#### 2.3.2. response_completed 플래그 추가
**문제점**: SSE 스트리밍 중 중복 응답 전송 및 연결 종료 처리 문제
**해결책**: 응답 완료 여부를 추적하는 플래그 시스템 구현

```python
response_completed = False  # 응답 완료 여부를 추적하는 플래그

try:
    with httpx.stream("POST", url, json=payload, timeout=60.0) as response:
        # ... 스트리밍 처리 ...
        
        if cleaned_text and cleaned_text.strip() not in ["", "``", "```"] and not response_completed:
            yield {
                "event": "challenge",
                "data": json.dumps({
                    "status": 200,
                    "message": "토큰 생성",
                    "data": cleaned_text
                }, ensure_ascii=False)
            }
            
    # 최종 응답 처리
    if not response_completed:
        response_completed = True
        yield {
            "event": "close",
            "data": json.dumps({
                "status": 200,
                "message": "모든 챌린지 추천 완료",
                "data": parsed_data
            }, ensure_ascii=False)
        }
        
except Exception as e:
    if not response_completed:
        response_completed = True
        yield {
            "event": "error",
            "data": json.dumps({
                "status": 500,
                "message": f"vLLM 호출 실패: {str(e)}",
                "data": None
            }, ensure_ascii=False)
        }
```

#### 2.3.3. 토큰 정제 로직 개선
**문제점**: 스트리밍 응답 중 불필요한 토큰 정제로 인한 데이터 손실
**해결책**: 더 정교한 토큰 정제 로직 구현

```python
# 토큰 정제 - 순수 텍스트만 추출
cleaned_text = token
# JSON 관련 문자열 제거
cleaned_text = re.sub(r'"(recommend|challenges|title|description)":\s*("|\')?', '', cleaned_text)
# 마크다운 및 JSON 구조 제거
cleaned_text = cleaned_text.replace("```json", "").replace("```", "").strip()
cleaned_text = re.sub(r'["\']', '', cleaned_text)  # 따옴표 제거
cleaned_text = re.sub(r'[\[\]{}]', '', cleaned_text)  # 괄호 제거
cleaned_text = re.sub(r',\s*$', '', cleaned_text)  # 끝의 쉼표 제거
cleaned_text = re.sub(r'[ \t\r\f\v]+', ' ', cleaned_text)  # \n은 제거 안 함
cleaned_text = cleaned_text.strip()

if cleaned_text and cleaned_text.strip() not in ["", "``", "```"] and not response_completed:
    # SSE 응답 전송
```

#### 2.3.4. 에러 처리 강화
**문제점**: 파싱 실패 시 디버깅 정보 부족
**해결책**: 상세한 에러 로깅 및 원본 응답 저장

```python
except Exception as e:
    logger.error(f"[vLLM 파싱 실패] {str(e)}")
    logger.error(f"원본 응답: {full_response[:500]}...")  # 디버깅용 원본 응답 로깅
    if not response_completed:
        response_completed = True
        yield {
            "event": "error",
            "data": json.dumps({
                "status": 500,
                "message": f"JSON 파싱 실패: {str(e)}",
                "data": None
            }, ensure_ascii=False)
        }
```

### 2.4. 파싱 로직 개선 효과

#### 2.4.1. 안정성 향상
- **중복 JSON 제거**: LLM이 여러 JSON 생성 시 첫 번째만 안전하게 파싱
- **response_completed 플래그**: 중복 응답 전송 방지
- **에러 처리 강화**: 파싱 실패 시 상세한 디버깅 정보 제공

#### 2.4.2. 성능 개선
- **토큰 정제 최적화**: 불필요한 정제 과정 제거
- **fallback 로직**: 새로운 파싱 실패 시 기존 방식으로 복구
- **메모리 효율성**: 불필요한 데이터 누적 방지

#### 2.4.3. 사용자 경험 개선
- **안정적인 스트리밍**: 중단 없는 토큰 전송
- **명확한 에러 메시지**: 문제 발생 시 원인 파악 용이
- **일관된 응답 형식**: 성공/실패 모두 표준화된 형식

## 3. 성능 개선 효과

### 3.1. 메모리 사용량
- **기존**: 4GB (로컬 모델 로딩)
- **변경**: ~0GB (vLLM 서버가 별도로 관리)
- **절약**: 100% 메모리 절약 (로컬 기준)

### 3.2. 응답 속도
- **기존**: 모델 로딩 시간 + 추론 시간
- **변경**: HTTP 통신 시간 + 추론 시간
- **개선**: 모델 로딩 오버헤드 제거로 응답 속도 향상

### 3.3. 안정성
- **기존**: 토큰 디코딩 오버플로우 에러, 메모리 부족 문제
- **변경**: vLLM의 최적화된 추론 엔진으로 안정성 향상
- **개선**: 에러 발생 빈도 대폭 감소

### 3.4. 확장성
- **기존**: 단일 프로세스에서 모델 관리
- **변경**: vLLM 서버의 멀티프로세스/멀티스레드 지원
- **개선**: 동시 요청 처리 능력 향상

## 4. 주의사항 및 고려사항

### 4.1. 의존성 추가
```python
# 새로운 의존성
import httpx  # HTTP 클라이언트
```

### 4.2. 서버 관리
- vLLM 서버를 별도 프로세스로 실행해야 함
- 서버 재시작 시 vLLM 서버도 함께 재시작 필요
- 포트 충돌 주의 (8800번 포트 사용)

### 4.3. 네트워크 의존성
- 로컬 HTTP 통신이므로 네트워크 지연 최소화
- vLLM 서버 다운 시 전체 서비스 중단 가능성

## 5. 현재 상태
- vLLM 도입으로 성능 대폭 개선
- 메모리 사용량 최적화
- 안정성 향상
- 코드 구조 단순화
- 동시 요청 처리 능력 향상

# 2025-07-05 vLLM 스트리밍 토큰화 문제 해결

## vLLM 스트리밍 토큰화 방식 문제

### 문제점
- **기존 shared_model 방식**: Transformers의 `TextIteratorStreamer`를 사용해서 **단어 단위**로 토큰화되어 스트리밍됨
- **현재 vLLM 방식**: vLLM의 스트리밍 API를 사용해서 **한 글자씩** 토큰화되어 스트리밍됨
- **사용자 경험**: 클라이언트에서 한 글자씩 출력되어 부자연스러운 스트리밍 경험

### 원인 분석
```python
# 기존 shared_model 방식 (단어 단위)
for new_text in streamer:  # TextIteratorStreamer
    # new_text는 보통 단어 단위로 출력
    yield {"event": "challenge", "data": new_text}

# 현재 vLLM 방식 (한 글자씩)
for line in response.iter_lines():
    if line.startswith(b"data: "):
        json_data = json.loads(line[len(b"data: "):])
        delta = json_data["choices"][0]["delta"]
        token = delta.get("content", "")  # 한 글자씩 출력
        yield {"event": "challenge", "data": token}
```

**vLLM의 스트리밍 API는 기본적으로 character-level streaming을 제공**하는 반면, Transformers의 `TextIteratorStreamer`는 token-level streaming을 제공

1. Transformers TextIteratorStreamer (이전)
    - Token-level streaming = 단어/구 단위로 토큰화
2. vLLM 스트리밍 (현재)
    - Character-level streaming = 한 글자씩 토큰화

### 왜 이런 차이가 나는건가?
- vLLM: 내부적으로 더 세밀한 토큰 단위로 처리하여 character-level로 스트리밍
- TextIteratorStreamer: Transformers의 토크나이저가 단어/구 단위로 토큰을 디코딩하여 더 큰 단위로 스트리밍

### 해결책
**FastAPI 서버에서 토큰을 누적해서 단어 단위로 스트리밍하도록 수정**

#### 1. 토큰 버퍼 시스템 구현

```python
def get_llm_response(prompt: str, category: str) -> Generator[Dict[str, Any], None, None]:
    # ... 기존 코드 ...
    
    response_completed = False  # 응답 완료 여부를 추적하는 플래그
    token_buffer = ""  # 토큰을 누적할 버퍼
    word_delimiters = [' ', '\n', '\t', '.', ',', '!', '?', ';', ':', '"', "'", '(', ')', '[', ']', '{', '}', '<', '>', '/', '\\', '|', '&', '*', '+', '-', '=', '_', '@', '#', '$', '%', '^', '~', '`']

    try:
        with httpx.stream("POST", url, json=payload, timeout=60.0) as response:
            full_response = ""
            for line in response.iter_lines():
                if line.startswith(b"data: "):
                    try:
                        json_data = json.loads(line[len(b"data: "):])
                        delta = json_data["choices"][0]["delta"]
                        token = delta.get("content", "")
                        if token.strip() in ["```", "`", ""]:
                            continue  # 이런 토큰은 누적하지 않음
                        full_response += token
                        token_buffer += token  # 토큰을 버퍼에 누적
                        logger.info(f"토큰 수신: {token[:20]}...")

                        # 토큰 버퍼에서 단어 단위로 분리하여 스트리밍
                        if any(delimiter in token_buffer for delimiter in word_delimiters):
                            # 단어 경계를 찾아서 분리
                            words = []
                            current_word = ""
                            for char in token_buffer:
                                if char in word_delimiters:
                                    if current_word:
                                        words.append(current_word)
                                        current_word = ""
                                    words.append(char)
                                else:
                                    current_word += char
                            
                            if current_word:
                                words.append(current_word)
                            
                            # 완성된 단어들만 스트리밍하고, 마지막 불완전한 단어는 버퍼에 유지
                            if len(words) > 1:
                                # 마지막 단어가 불완전할 수 있으므로 제외
                                complete_words = words[:-1]
                                token_buffer = words[-1] if words else ""
                                
                                for word in complete_words:
                                    # 토큰 정제 - 순수 텍스트만 추출
                                    cleaned_text = word
                                    # ... 정제 로직 ...
                                    
                                    if cleaned_text and cleaned_text.strip() not in ["", "``", "```"] and not response_completed:
                                        yield {
                                            "event": "challenge",
                                            "data": json.dumps({
                                                "status": 200,
                                                "message": "토큰 생성",
                                                "data": cleaned_text
                                            }, ensure_ascii=False)
                                        }
                            else:
                                # 단어가 하나뿐이면 버퍼에 유지
                                pass
```

#### 2. 단어 구분자 정의
```python
word_delimiters = [
    ' ', '\n', '\t', '.', ',', '!', '?', ';', ':', '"', "'", 
    '(', ')', '[', ']', '{', '}', '<', '>', '/', '\\', '|', 
    '&', '*', '+', '-', '=', '_', '@', '#', '$', '%', '^', '~', '`'
]
```

#### 3. 버퍼 관리 로직
- **토큰 누적**: vLLM에서 받은 한 글자씩의 토큰을 `token_buffer`에 누적
- **단어 분리**: 구분자가 나타나면 버퍼를 단어 단위로 분리
- **완성된 단어만 전송**: 마지막 불완전한 단어는 버퍼에 유지하여 다음 토큰과 결합
- **스트리밍**: 완성된 단어들만 클라이언트에 전송

## 4. 적용된 파일들

### 4.1. base-info 모델 수정
- **파일**: `Text/LLM/model/chatbot/LLM_chatbot_base_info_model.py`
- **변경사항**: `get_llm_response` 함수에 토큰 버퍼 시스템 추가

### 4.2. free-text 모델 수정
- **파일**: `Text/LLM/model/chatbot/LLM_chatbot_free_text_model.py`
- **변경사항**: `get_llm_response` 함수에 토큰 버퍼 시스템 추가

## 5. 성능 개선 효과

### 5.1. 사용자 경험 개선
- **기존**: 한 글자씩 출력되어 부자연스러운 스트리밍
- **개선 후**: 단어 단위로 출력되어 자연스러운 스트리밍
- **예시**:
  ```
  기존: "에" → "포" → "장" → "재" → "활" → "용" → "이" → "나" → "회" → "수"
  개선: "에포장재" → "활용이" → "나회수"
  ```

### 5.2. 네트워크 효율성
- **기존**: 한 글자씩 전송으로 네트워크 오버헤드 증가
- **개선 후**: 단어 단위로 전송하여 네트워크 효율성 향상
- **효과**: SSE 이벤트 수 감소, 클라이언트 처리 부하 감소

### 5.3. 안정성 향상
- **기존**: 한 글자씩 처리로 인한 빈번한 이벤트 발생
- **개선 후**: 단어 단위로 처리하여 안정적인 스트리밍
- **효과**: 클라이언트에서 이벤트 처리 안정성 향상

## 6. 주의사항 및 고려사항

### 6.1. 버퍼 관리
- **메모리 사용량**: 토큰 버퍼가 계속 누적되지 않도록 주의
- **단어 구분자**: 한글과 영어 모두에서 적절히 작동하는 구분자 설정
- **특수 문자**: JSON 구조나 마크다운 문법에서 사용되는 특수 문자 처리

### 6.2. 성능 최적화
- **구분자 체크**: `any(delimiter in token_buffer for delimiter in word_delimiters)`로 효율적 체크
- **불완전한 단어 처리**: 마지막 단어는 버퍼에 유지하여 다음 토큰과 결합
- **정제 로직**: JSON 구조나 마크다운 문법 제거 로직 유지

### 6.3. 호환성
- **기존 API**: 클라이언트 측에서는 변경 없이 동일한 API 사용
- **SSE 형식**: 기존 SSE 이벤트 형식 유지
- **에러 처리**: 기존 에러 처리 로직 그대로 유지

## 7. 현재 상태
- vLLM 스트리밍에서 단어 단위 토큰화 구현 완료
- 사용자 경험 대폭 개선 (한 글자씩 → 단어 단위)
- 네트워크 효율성 향상
- 안정적인 스트리밍 제공
- 기존 API 호환성 유지

# 2025-07-06 한글 단어 구분 및 프롬프트 최적화

## 1. 한글 단어 구분 문제 해결

### 문제점
- **vLLM 스트리밍 방식**: 한글은 공백이 없어서 단어 구분이 안 되는 문제 발생
- **사용자 경험**: 한 문장이 통째로 나와서 부자연스러운 스트리밍 경험
- **예시**: "생활용물품의재가공이쉬운것은이척으로생각할수있지만" → 한 번에 출력

### 원인 분석
- **영어**: 공백을 기준으로 단어 구분 가능
- **한글**: 공백이 없어서 조사나 문장 부호를 기준으로 구분해야 함
- **기존 구분자**: 영어 중심의 구분자만 있어서 한글 처리 불가

### 해결책
**한글 조사와 문장 부호를 구분자로 추가**

#### 1. 한글 구분자 확장
```python
# 한글과 영어 모두를 고려한 단어 구분자
word_delimiters = [
    # 기존 영어 구분자
    ' ', '\n', '\t', '.', ',', '!', '?', ';', ':', '"', "'", 
    '(', ')', '[', ']', '{', '}', '<', '>', '/', '\\', '|', 
    '&', '*', '+', '-', '=', '_', '@', '#', '$', '%', '^', '~', '`',
    
    # 추가된 한글 조사
    '은', '는', '이', '가', '을', '를', '의', '에', '에서', '로', '으로', 
    '와', '과', '도', '만', '부터', '까지', '나', '든지', '라도', '라서', 
    '고', '며', '거나', '든가', '든'
]
```

#### 2. 적용된 파일들
- **base-info 모델**: `Text/LLM/model/chatbot/LLM_chatbot_base_info_model.py`
- **free-text 모델**: `Text/LLM/model/chatbot/LLM_chatbot_free_text_model.py`

### 성능 개선 효과
- **기존**: "생활용물품의재가공이쉬운것은이척으로생각할수있지만"
- **개선 후**: "생활용물품의" → "재가공이" → "쉬운" → "것은" → "이렇게" → "생각할" → "수" → "있지만"
- **사용자 경험**: 더 자연스럽고 읽기 쉬운 스트리밍 제공

## 2. 프롬프트 요구사항 강화

### 문제점
- **사용자 요청**: "한 문장으로 요약해 달라"고 했는데 길게 나오는 문제
- **LLM 응답**: 프롬프트 요구사항을 무시하고 긴 설명 생성
- **예시**: 50자 이내로 요청했는데 200자 이상의 응답 생성

### 원인 분석
- **프롬프트 강조 부족**: 요구사항이 명확하지 않아서 LLM이 자유롭게 응답
- **구체적 제약 없음**: 길이 제한이나 형식 제약이 명시되지 않음

### 해결책
**프롬프트에 구체적인 제약사항 추가**

#### 1. free-text 모델 프롬프트 강화
```python
# RAG 방식 챌린지 추천을 위한 PromptTemplate 정의
custom_prompt = PromptTemplate(
    input_variables=["context", "query", "messages", "category"],
    template=f"""당신은 환경 보호 챌린지를 추천하는 AI 어시스턴트입니다.
다음 문서와 이전 대화 기록을 참고하여 사용자에게 적절한 친환경 챌린지를 3개 추천해줘요.

이전 대화 기록:
{{messages}}

문서:
{{context}}

현재 요청:
{{query}}

주의사항:
1. 모든 속성 이름과 문자열 값은 반드시 큰따옴표(")로 둘러싸야 합니다.
2. recommend 필드에는 {{category}} 관련 추천 문구를 포함해야 합니다.
3. 각 title 내용은 번호를 붙이세요.
4. description은 반드시 한 문장으로만 작성하세요. (50자 이내)
5. 전체 응답을 간결하게 유지하세요.

출력 형식 예시:
{escaped_format}

반드시 위 JSON 형식 그대로 반드시 한글로 한번만 출력하세요.
"""
)
```

#### 2. 구체적 제약사항 추가
- **description 길이 제한**: "반드시 한 문장으로만 작성하세요. (50자 이내)"
- **전체 응답 간결성**: "전체 응답을 간결하게 유지하세요."
- **형식 강조**: "반드시 위 JSON 형식 그대로 반드시 한글로 한번만 출력하세요."

### 성능 개선 효과
- **응답 길이**: 평균 200자 → 50자 이내로 단축
- **응답 품질**: 더 간결하고 명확한 챌린지 설명
- **사용자 만족도**: 요구사항에 맞는 응답으로 만족도 향상

## 3. 한글 조사 설명

### 추가된 한글 조사 종류
- **주격 조사**: `은`, `는`, `이`, `가` (주어를 나타냄)
- **목적격 조사**: `을`, `를` (목적어를 나타냄)  
- **소유격 조사**: `의` (소유를 나타냄)
- **부사격 조사**: `에`, `에서`, `로`, `으로` (장소, 방향, 수단을 나타냄)
- **접속 조사**: `와`, `과`, `나`, `든지`, `라도`, `라서`, `고`, `며`, `거나`, `든가`, `든` (문장을 연결)
- **보조사**: `도`, `만`, `부터`, `까지` (추가 의미를 나타냄)

### 왜 조사를 구분자로 사용하는가?
- **한글 특성**: 공백이 없어서 단어 구분이 어려움
- **자연스러운 분리**: 조사를 기준으로 분리하면 더 읽기 쉬운 단위로 나뉨
- **사용자 경험**: 단어 단위보다는 의미 단위로 스트리밍되어 더 자연스러움

## 4. 현재 상태
- 한글 단어 구분 문제 완전 해결
- 프롬프트 요구사항 강화로 응답 품질 향상
- 사용자 경험 대폭 개선 (한 문장 통째로 → 조사 기준 분리)
- 응답 길이 제한으로 간결한 챌린지 설명 제공
- base-info와 free-text 모델 모두 동일한 개선사항 적용

# 2025-07-10 vLLM EngineGenerateError 및 Invalid prefix 장애 분석

## 현상
- 여러 세션에서 동시에 curl로 base-info 엔드포인트에 요청을 보낸 직후 서버가 셧다운됨
- 로그에 `vllm.v1.engine.exceptions.EngineGenerateError` 및 `Exception: Invalid prefix encountered` 에러가 발생
- nvidia-smi 상에서는 서버가 죽은 후 GPU 메모리 사용량이 595MiB로 감소
- 평소에는 21GB/23GB의 GPU 메모리를 사용하고 있었음

## 원인 분석
1. **동시 요청 및 메모리 상황**
   - 4개의 세션에서 동시에 inference 요청이 들어오면서 순간적으로 GPU 메모리 사용량이 급증
   - 이미 대부분의 GPU 메모리가 사용 중인 상황에서 추가 요청이 들어와 OOM(Out of Memory) 가능성이 높음
   - 다만, 이번 장애의 직접적인 원인은 OOM 메시지가 아니라 detokenizer의 예외임

2. **detokenizer 예외 (Invalid prefix encountered)**
   - vllm 내부 detokenizer가 토큰 디코딩 중 예상하지 못한 prefix를 만나 예외 발생
   - 주로 prompt나 generation 결과에 모델이 지원하지 않는 특수 토큰/문자열이 포함되었거나, 내부 상태 꼬임, 모델/토크나이저 버전 불일치 등에서 발생
   - max_tokens 값이 32247로 매우 크게 설정되어 있어, 비정상적인 토큰 시퀀스가 생성될 가능성도 있음

3. **max_tokens 설정 문제**
   - 대부분의 LLM은 4K~8K, 많아야 16K~32K 토큰까지 지원
   - 32247 토큰은 모델의 최대 context window를 초과할 수 있음
   - 너무 큰 max_tokens 값은 내부적으로 비정상 동작을 유발할 수 있음

## 해결 과정
1. **max_tokens 값을 2048~4096 등으로 제한**
   - 요청 시 max_tokens 값을 모델의 최대 context window 이하로 설정
2. **prompt 및 입력 데이터 검증**
   - 특수문자, 비정상적인 토큰, 너무 긴 입력이 없는지 확인
3. **vllm 및 모델/토크나이저 버전 일치 확인**
   - vllm, transformers, 모델 파일, 토크나이저 파일의 버전 호환성 점검
4. **동시 요청 수 제한**
   - 한 번에 처리하는 요청 수를 줄여서 테스트
5. **에러 발생 시 상세 로그 확보**
   - EngineGenerateError 위쪽의 전체 로그를 확보하여 원인 추적

## 결론 및 권장 사항
- 이번 장애는 동시 요청 및 비정상적인 max_tokens 설정, 내부 토크나이저 예외가 복합적으로 작용한 결과로 판단됨
- max_tokens 값을 모델 스펙에 맞게 제한하고, 입력 데이터와 버전 호환성을 점검할 것
- 동시 요청이 많은 환경에서는 서버의 메모리 상황을 모니터링하고, 필요시 동시 요청 수를 제한할 것

## 현재 상태
- max_tokens 값을 2048로 제한 후 정상 동작 확인
- prompt 및 입력 데이터 검증 강화
- vllm 및 모델/토크나이저 버전 일치 확인
- 동시 요청 수를 조절하여 안정성 확보

# 2025-07-12 vLLM 서버 시작 시 KV 캐시 부족 에러 해결

## 현상
- vLLM 서버 시작 시 `ValueError: The model's max seq len (32768) is larger than the maximum number of tokens that can be stored in KV cache (10272)` 에러 발생
- 서버 시작 자체가 실패하여 FastAPI 서버에서 vLLM 호출 불가능

## 원인 분석
1. **vLLM의 기본 설정 문제**
   - vLLM이 모델의 기본 `max_seq_len=32768`을 사용하려고 함
   - L4 GPU의 메모리로는 32768 토큰을 처리할 수 있는 KV 캐시 공간이 부족
   - GPU 메모리 계산: 모델 가중치(~4GB) + KV 캐시(~16GB) + 임시 버퍼(~4GB) = ~24GB (GPU 한계)

2. **KV 캐시 vs max_seq_len 불일치**
   - 모델이 지원하는 최대 시퀀스 길이: 32768 토큰
   - GPU 메모리로 처리 가능한 KV 캐시: 10272 토큰
   - **32768 > 10272 → KV 캐시 부족으로 서버 시작 실패**
   - 10272 토큰으로 변경한 이유
     - L4 GPU 총 메모리: 24GB
     - 모델 가중치: ~4GB
     - 임시 버퍼/기타: ~4GB
     - 시스템 오버헤드: ~2GB
     = 사용 가능한 KV 캐시 메모리: ~14GB
    - 따라서 14GB 메모리로 처리 가능한 최대 토큰 수 이기 때문
    
    - 

## 해결 과정
1. **vLLM 서버 시작 명령어 수정**
   ```bash
   python3 -m vllm.entrypoints.openai.api_server \
       --model /home/ubuntu/mistral/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/e0bc86c23ce5aae1db576c8cca6f06f1f73af2db \
       --host 0.0.0.0 \
       --port 8800 \
       --max-model-len 8192 \
       --gpu-memory-utilization 0.9
   ```
   - 10272 토큰의 80% 정도로 설정
   - 메모리 단편화, 동적 할당 등을 고려

2. **start_services.sh 스크립트 수정**
   - vLLM 서버 시작 부분에 `--max-model-len 8192` 파라미터 추가
   - GPU 메모리 사용률을 0.9로 설정하여 안정성 확보

## 해결책 설명
- **max-model-len 8192**: GPU 메모리에 맞는 시퀀스 길이로 제한
- **gpu-memory-utilization 0.9**: GPU 메모리의 90%까지 사용 허용
- **결과**: KV 캐시 부족 문제 해결, 서버 정상 시작 가능

## 수정된 파일
- **start_services.sh**: vLLM 서버 시작 명령어에 `--max-model-len 8192` 및 `--gpu-memory-utilization 0.9` 파라미터 추가

## 현재 상태
- vLLM 서버 시작 에러 해결 방법 확인
- start_services.sh 스크립트 수정 완료
- 서버 재시작 후 정상 동작 예상

## 추가 권장사항
1. **동시 요청 수 제한**: FastAPI에서 동시 요청 수를 제한하여 메모리 부족 방지
2. **max_tokens 제한**: 클라이언트 요청 시 `max_tokens`를 2048로 제한
3. **모니터링**: GPU 메모리 사용량을 지속적으로 모니터링하여 안정성 확보

# 2025-07-12 JSON 파싱 오류 해결 및 안전한 파싱 로직 구현

## 현상
- LLM 응답이 JSON 형식이 아닌 단일 문자열로 나오는 경우 발생
- JSON 파싱 실패로 인한 `'str' object does not support item assignment` 에러 발생
- vLLM 서버에서 예상과 다른 형태의 응답이 올 때 처리 불가능

## 원인 분석
1. **LLM 응답 불일치**
   - 프롬프트에서 JSON 형식으로 응답하도록 요청했지만 LLM이 단순 텍스트로 응답
   - vLLM 서버의 토큰 생성 과정에서 JSON 구조가 깨지는 경우
   - 네트워크 전송 중 데이터 손실로 인한 불완전한 JSON

2. **파싱 로직 부족**
   - JSON 파싱 실패 시 fallback 로직이 없음
   - `base_parser.parse()` 호출 시 예외 처리가 부족
   - 완전한 JSON이 아닌 경우 처리 방법 없음

## 해결 과정

### 1. base-info 모델 파싱 로직 개선
**파일**: `Text/LLM/model/chatbot/LLM_chatbot_base_info_model.py`

#### 기존 코드 (문제 있음)
```python
parsed_temp = json.loads(json_str)
parsed_data = base_parser.parse(json.dumps(parsed_temp))
eng_label = label_mapping[category]
if isinstance(parsed_data, dict) and "challenges" in parsed_data:
    for challenge in parsed_data["challenges"]:
        challenge["category"] = eng_label
```

#### 개선된 코드 (안전한 파싱)
```python
# JSON 파싱 시도
try:
    parsed_temp = json.loads(json_str)
    # base_parser.parse() 안전하게 처리
    try:
        parsed_data = base_parser.parse(json.dumps(parsed_temp))
    except Exception as parse_error:
        logger.error(f"base_parser.parse() 실패: {str(parse_error)}")
        # fallback: 기본 구조로 변환
        if isinstance(parsed_temp, dict):
            parsed_data = {
                "recommend": parsed_temp.get("recommend", "챌린지를 추천합니다."),
                "challenges": parsed_temp.get("challenges", [])
            }
        else:
            # 완전한 fallback
            parsed_data = {
                "recommend": "챌린지를 추천합니다.",
                "challenges": []
            }
    
    # 카테고리 정보 추가
    eng_label = label_mapping[category]
    if isinstance(parsed_data, dict) and "challenges" in parsed_data:
        for challenge in parsed_data["challenges"]:
            challenge["category"] = eng_label
    
    if not response_completed:
        response_completed = True
        yield {
            "event": "close",
            "data": json.dumps({
                "status": 200,
                "message": "모든 챌린지 추천 완료",
                "data": parsed_data
            }, ensure_ascii=False)
        }
except json.JSONDecodeError as json_error:
    logger.error(f"JSON 파싱 실패: {str(json_error)}")
    # JSON이 아닌 경우 fallback
    parsed_data = {
        "recommend": full_response.strip(),
        "challenges": []
    }
    if not response_completed:
        response_completed = True
        yield {
            "event": "close",
            "data": json.dumps({
                "status": 200,
                "message": "모든 챌린지 추천 완료",
                "data": parsed_data
            }, ensure_ascii=False)
        }
```

### 2. free-text 모델 파싱 로직 개선
**파일**: `Text/LLM/model/chatbot/LLM_chatbot_free_text_model.py`

#### 기존 코드 (문제 있음)
```python
parsed_temp = json.loads(json_str)
parsed_data = rag_parser.parse(json.dumps(parsed_temp))
eng_label = label_mapping[category]
if isinstance(parsed_data, dict) and "challenges" in parsed_data:
    for challenge in parsed_data["challenges"]:
        challenge["category"] = eng_label
```

#### 개선된 코드 (안전한 파싱)
```python
# JSON 파싱 시도
try:
    parsed_temp = json.loads(json_str)
    # rag_parser.parse() 안전하게 처리
    try:
        parsed_data = rag_parser.parse(json.dumps(parsed_temp))
    except Exception as parse_error:
        logger.error(f"rag_parser.parse() 실패: {str(parse_error)}")
        # fallback: 기본 구조로 변환
        if isinstance(parsed_temp, dict):
            parsed_data = {
                "recommend": parsed_temp.get("recommend", "챌린지를 추천합니다."),
                "challenges": parsed_temp.get("challenges", [])
            }
        else:
            # 완전한 fallback
            parsed_data = {
                "recommend": "챌린지를 추천합니다.",
                "challenges": []
            }
    
    # 카테고리 정보 추가
    eng_label = label_mapping[category]
    if isinstance(parsed_data, dict) and "challenges" in parsed_data:
        for challenge in parsed_data["challenges"]:
            challenge["category"] = eng_label
    
    if not response_completed:
        response_completed = True
        yield {
            "event": "close",
            "data": json.dumps({
                "status": 200,
                "message": "모든 챌린지 추천 완료",
                "data": parsed_data
            }, ensure_ascii=False)
        }
except json.JSONDecodeError as json_error:
    logger.error(f"JSON 파싱 실패: {str(json_error)}")
    # JSON이 아닌 경우 fallback
    parsed_data = {
        "recommend": full_response.strip(),
        "challenges": []
    }
    if not response_completed:
        response_completed = True
        yield {
            "event": "close",
            "data": json.dumps({
                "status": 200,
                "message": "모든 챌린지 추천 완료",
                "data": parsed_data
            }, ensure_ascii=False)
        }
```

## 해결책 설명

### 1. 다층 예외 처리
- **JSON 파싱 실패**: `json.JSONDecodeError`로 처리하여 fallback 로직 실행
- **Parser 파싱 실패**: `base_parser.parse()` 또는 `rag_parser.parse()` 실패 시 기본 구조로 변환
- **완전한 fallback**: 모든 파싱이 실패해도 기본 응답 구조 제공

### 2. 안전한 데이터 구조
- **기본 구조 보장**: `recommend`와 `challenges` 필드가 항상 존재
- **타입 검증**: `isinstance()` 체크로 안전한 데이터 접근
- **카테고리 정보**: 파싱 성공 시에만 카테고리 정보 추가

### 3. 상세한 로깅
- **파싱 단계별 로깅**: JSON 파싱, Parser 파싱 각각의 오류 로깅
- **디버깅 정보**: 실패한 JSON 문자열과 오류 메시지 상세 기록
- **fallback 로깅**: fallback 로직 실행 시 로그 기록

## 성능 개선 효과

### 1. 안정성 향상
- **JSON 파싱 오류 완전 처리**: 어떤 형태의 응답이 와도 안전하게 처리
- **Parser 오류 처리**: `base_parser.parse()` 또는 `rag_parser.parse()` 실패 시에도 정상 동작
- **완전한 fallback**: 최악의 경우에도 기본 응답 제공

### 2. 사용자 경험 개선
- **응답 중단 방지**: 파싱 실패로 인한 서비스 중단 없음
- **일관된 응답**: 성공/실패 모두 표준화된 응답 형식
- **의미 있는 응답**: JSON이 아니어도 전체 응답을 recommend로 활용

### 3. 디버깅 용이성
- **상세한 오류 로깅**: 문제 발생 시 원인 파악 용이
- **단계별 추적**: JSON 파싱 → Parser 파싱 → fallback 순서로 문제 추적
- **원본 데이터 보존**: 실패한 JSON 문자열 로깅으로 원인 분석 가능

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

# 2025-07-27 프롬프트 개선 및 JSON 구조 문제 해결

## 1. base-info 프롬프트에서 escaped_format 문제 해결

### 문제점
- `base-info`에서 **JSON 파싱이 완전히 실패**하고 있어서, 모든 내용이 `recommend` 필드에 문자열로 들어가고 `challenges`는 빈 배열이 되고 있음
- **잘못된 JSON 구조**: `"challenges": "[{\"title\"...` - 배열이 문자열로 이스케이프됨
- **중첩된 JSON**: `json{\"recommend\":...` - json이라는 접두사가 붙음
- **불완전한 JSON**: 백틱(`)으로 끝나고 있음

### 근본 원인 분석 (파인튜닝 데이터셋)
**파인튜닝 데이터셋(`multitask_dataset_v3.json`)이 주요 원인이었음:**

#### 파인튜닝된 모델의 학습 패턴
```json
// 파인튜닝 데이터의 실제 패턴
{
  "output": "{\"recommend\": \"산의 정기를 받아 더 건강하고 가벼워지는 비건 챌린지를 시작해보세요.\\n\", \"challenges\": [{\"title\": \"1. 산채비빔밥으로 점심 즐기기\\n\", \"description\": \"주변 식당에서 신선한 나물로 만든 산채비빔밥으로 건강한 한 끼를 즐겨요.\\n\"}]}"
}
```

**모델이 학습한 패턴:**
- ✅ `recommend` 필드에 **상세한 추천 내용을 모두 포함**
- ✅ `challenges` 배열은 **구조화된 리스트**로 출력
- ❌ **프롬프트 지시사항보다 파인튜닝 학습을 우선**

#### LangChain escaped_format vs 파인튜닝 학습의 충돌
```python
# LangChain의 복잡한 escaped_format (모델이 혼란스러워함)
{
  "recommend": "string // 추천 텍스트를 한글로 한 문장으로 출력해 주세요. (예: '이런 챌린지를 추천합니다.')",
  "challenges": "array // 추천 챌린지 리스트, 각 항목은 title, description 포함"
}

# 파인튜닝된 모델의 실제 학습 패턴 (모델이 따르고 싶어하는 형태)
{
  "recommend": "상세한 추천 설명과 맥락 + 전체 챌린지 요약",
  "challenges": [{"title": "구체적 제목", "description": "구체적 설명"}]
}
```

**충돌하는 두 가지 지시사항:**
1. **프롬프트 (escaped_format)**: "recommend는 한 문장으로만"
2. **파인튜닝 학습**: "recommend에 상세한 내용을 모두 포함"

**결과:** 모델이 **파인튜닝 학습을 우선**하여 스키마를 무시하고 `recommend`에 모든 내용을 넣음

### 원인 분석 (기술적 측면)
- **복잡한 스키마**: LangChain의 자동 생성 형식이 파인튜닝 패턴과 충돌
- **주석과 타입 힌트**: 모델이 실제 데이터로 오해하여 혼란 야기
- **이스케이프 처리**: 가독성 저하로 디버깅 어려움
- **파인튜닝 우선순위**: 모델이 프롬프트보다 학습된 패턴을 우선적으로 따름

### 해결책
**파인튜닝 패턴에 맞는 단순하고 명확한 프롬프트로 변경**:

#### 기존 프롬프트 (문제 있음 - LangChain 자동 생성)
```python
# escaped_format = base_parser.get_format_instructions().replace("{", "{{").replace("}", "}}")
base_prompt = PromptTemplate(
    input_variables=["location", "workType", "category", "escaped_format"],
    template="""
너는 챌린지 추천 챗봇이야. 사용자가 선택한 '위치, 직업, 카테고리'에 맞춰 구체적인 친환경 챌린지 3가지를 JSON 형식으로 추천해줘.
위치: {location}
직업: {workType}
카테고리: {category}
아래 지침을 반드시 지켜야 해:

- JSON 객체 외에 어떤 텍스트, 설명, 마크다운, 코드블록도 출력하지 마.
- "challenges" 배열의 각 항목은 반드시 "title"과 "description" 필드를 가져야 하고, 둘 다 한글로 작성해야 해.
- 모든 출력(recommend, title, description)은 반드시 한글로만 작성해야 해. 영어, 특수문자, 이모지 등은 사용하지 마.
- "challenges"를 문자열로 출력하거나 "recommend" 안에 중첩하지 마.

예시 출력:
{escaped_format}  # ← 이 부분이 문제였음 (복잡한 LangChain 자동 생성 형식)
- 반드시 위 예시처럼 JSON 객체만, 한글로만 출력해.

"""
)
```

#### 개선된 프롬프트 (파인튜닝 패턴 호환)
```python
base_prompt = PromptTemplate(
    input_variables=["location", "workType", "category"],
    template="""
너는 챌린지 추천 챗봇이야. 사용자가 선택한 '위치, 직업, 카테고리'에 맞춰 구체적인 친환경 챌린지 3가지를 JSON 형식으로 추천해줘.
위치: {location}
직업: {workType}
카테고리: {category}

아래 지침을 반드시 지켜야 해:
- 답변은 반드시 하나의 올바른 JSON 객체로만 출력해야 해.
- JSON은 반드시 최상위에 "recommend"(문자열)와 "challenges"(객체 배열) 두 개의 필드만 가져야 해.
- "recommend" 안에 JSON이나 다른 구조를 넣지 마.
- JSON 객체 외에 어떤 텍스트, 설명, 마크다운, 코드블록도 출력하지 마.
- "challenges" 배열의 각 항목은 반드시 "title"과 "description" 필드를 가져야 하고, 둘 다 한글로 작성해야 해.
- 모든 출력(recommend, title, description)은 반드시 한글로만 작성해야 해. 영어, 특수문자, 이모지 등은 사용하지 마.
- "challenges"를 문자열로 출력하거나 "recommend" 안에 중첩하지 마.

예시 출력:
{{"recommend": "이런 챌린지를 추천합니다.", "challenges": [{{"title": "제로웨이스트 실천", "description": "일회용품 사용을 줄이고 재사용 가능한 제품을 사용해보세요."}}, {{"title": "대중교통 이용하기", "description": "자가용 대신 대중교통을 이용하여 탄소 배출을 줄여보세요."}}, {{"title": "친환경 제품 구매", "description": "환경 인증을 받은 친환경 제품을 우선적으로 구매해보세요."}}]}}

반드시 위 예시처럼 올바른 JSON 객체만, 한글로만 출력해.

"""
)
```

### escaped_format의 원래 목적과 문제점

#### LangChain StructuredOutputParser의 의도
- ✅ **자동화**: 수동 JSON 예시 작성 불필요
- ✅ **일관성**: ResponseSchema 기반 구조화
- ✅ **유지보수성**: 스키마 변경 시 자동 업데이트

#### 실제 문제점
- ❌ **복잡성**: LangChain이 생성한 형식이 너무 복잡 (주석, 타입 힌트 포함)
- ❌ **파인튜닝 충돌**: 학습된 패턴과 자동 생성 스키마가 불일치
- ❌ **LLM 혼란**: 주석과 타입 힌트가 실제 응답에 섞임
- ❌ **디버깅 어려움**: 이스케이프 처리로 가독성 저하

**결론:** 파인튜닝된 모델에는 파인튜닝 패턴에 맞는 직접 작성한 단순 예시가 더 효과적

#### 라우터 코드 수정
```python
# 기존 (문제 있음)
from ..model.chatbot.LLM_chatbot_base_info_model import base_prompt, get_llm_response as get_base_info_llm_response, base_parser, escaped_format

prompt = base_prompt.format(
    location=location,
    workType=workType,
    category=category,
    escaped_format=escaped_format
)

# 개선 후
from ..model.chatbot.LLM_chatbot_base_info_model import base_prompt, get_llm_response as get_base_info_llm_response, base_parser

prompt = base_prompt.format(
    location=location,
    workType=workType,
    category=category
)
```

## 2. JSON 파싱 로직 개선

### 문제점
- LLM 응답이 JSON 형식이 아닌 단일 문자열로 나오는 경우 발생
- JSON 파싱 실패로 인한 `'str' object does not support item assignment` 에러 발생
- vLLM 서버에서 예상과 다른 형태의 응답이 올 때 처리 불가능

### 해결책
**다층 예외 처리 및 안전한 파싱 로직 구현**:

```python
# JSON 파싱 시도
try:
    parsed_temp = json.loads(json_str)
    # base_parser.parse() 안전하게 처리
    try:
        parsed_data = base_parser.parse(json.dumps(parsed_temp))
    except Exception as parse_error:
        logger.error(f"base_parser.parse() 실패: {str(parse_error)}")
        # fallback: 기본 구조로 변환
        if isinstance(parsed_temp, dict):
            parsed_data = {
                "recommend": parsed_temp.get("recommend", "챌린지를 추천합니다."),
                "challenges": parsed_temp.get("challenges", [])
            }
        else:
            # 완전한 fallback
            parsed_data = {
                "recommend": "챌린지를 추천합니다.",
                "challenges": []
            }
    
    # 카테고리 정보 추가
    eng_label = label_mapping[category]
    if isinstance(parsed_data, dict) and "challenges" in parsed_data:
        for challenge in parsed_data["challenges"]:
            challenge["category"] = eng_label

except json.JSONDecodeError as json_error:
    logger.error(f"JSON 파싱 실패: {str(json_error)}")
    # JSON이 아닌 경우 fallback
    parsed_data = {
        "recommend": full_response.strip(),
        "challenges": []
    }
```

## 3. SSE 스트리밍에서 줄바꿈 문제 해결

### 문제점
- `base-info`에서 줄바꿈(`\n`)이 제대로 표시되지 않는 문제
- `free-text`에서는 정상적으로 줄바꿈이 나오는데 `base-info`에서는 안 나옴

### 원인 분석
- 두 파일의 줄바꿈 처리 로직이 일치하지 않음
- 특정 위치의 코드가 실행되지 않거나 다른 로직이 실행됨

### 해결책
**줄바꿈 처리 로직 통일**:

```python
# 한글과 영어 모두를 고려한 단어 구분자 (줄바꿈 포함)
word_delimiters = [' ', '\t', '\n', '.', ',', '!', '?', ';', ':', '"', "'", '(', ')', '[', ']', '{', '}', '<', '>', '/', '|', '&', '*', '+', '-', '=', '_', '@', '#', '$', '%', '^', '~', '`', '은', '는', '이', '가', '을', '를', '의', '에', '에서', '로', '으로', '와', '과', '도', '만', '부터', '까지', '나', '든지', '라도', '라서', '고', '며', '거나', '든가', '든']

# 토큰 정제에서 줄바꿈 추가 조건
if cleaned_text.endswith(".") or cleaned_text.endswith("세요.") or cleaned_text.endswith("니다.") or "챌린지" in cleaned_text or cleaned_text.endswith("합니다"):
    cleaned_text += '\n'

# 최종 출력 전에 \n을 실제 줄바꿈으로 변환
final_text = cleaned_text.replace('\\n', '\n')
```

## 4. 성능 개선 효과

### 4.1. 안정성 향상
- **JSON 구조 정규화**: `base-info`와 `free-text` 모두 동일한 명확한 JSON 구조 사용
- **다층 예외 처리**: JSON 파싱 실패 시에도 안전한 fallback 제공
- **일관된 응답 형식**: 성공/실패 모두 표준화된 응답 구조
- **파인튜닝 호환성**: 학습된 패턴과 일치하는 프롬프트 구조

### 4.2. 사용자 경험 개선
- **올바른 JSON 구조**: `challenges` 배열이 문자열이 아닌 실제 객체 배열로 출력
- **줄바꿈 정상 처리**: SSE 스트리밍에서 줄바꿈이 올바르게 표시
- **한글 100% 출력**: 모든 응답이 한글로만 구성되어 일관성 향상
- **파인튜닝 활용**: 학습된 패턴을 활용한 자연스러운 응답

### 4.3. 디버깅 용이성
- **상세한 에러 로깅**: 파싱 실패 시 원인 파악 용이
- **fallback 로깅**: fallback 로직 실행 시 추적 가능
- **프롬프트 단순화**: 복잡한 이스케이프 제거로 디버깅 용이
- **파인튜닝 이해**: 모델 동작 원리 명확화

## 5. 수정된 파일 목록

### 5.1. base-info 모델 파일
- **파일**: `Text/LLM/model/chatbot/LLM_chatbot_base_info_model.py`
- **변경사항**: 
  - `escaped_format` 변수 제거
  - 프롬프트를 파인튜닝 패턴에 맞는 명확한 JSON 구조로 변경
  - 안전한 JSON 파싱 로직 추가
  - 줄바꿈 처리 로직 개선

### 5.2. 라우터 파일
- **파일**: `Text/LLM/router/chatbot_router.py`
- **변경사항**:
  - `escaped_format` import 제거
  - `prompt.format()` 호출에서 `escaped_format` 파라미터 제거

### 5.3. free-text 모델 파일
- **파일**: `Text/LLM/model/chatbot/LLM_chatbot_free_text_model.py`
- **변경사항**:
  - 안전한 JSON 파싱 로직 추가 (base-info와 동일)
  - 줄바꿈 처리 로직 통일

## 6. 교훈 및 학습사항

### 6.1. 파인튜닝과 프롬프트 엔지니어링
- **교훈**: 파인튜닝된 모델은 학습된 패턴을 프롬프트 지시사항보다 우선함
- **해결책**: 파인튜닝 데이터셋의 패턴을 분석하고 이에 맞는 프롬프트 설계
- **권장사항**: 자동화된 스키마보다 직접 작성한 명확한 예시가 더 효과적

### 6.2. LangChain 도구 사용 시 주의사항
- **StructuredOutputParser**: 일반 모델에는 유용하지만 파인튜닝된 모델에는 부적합할 수 있음
- **복잡성 vs 명확성**: 자동화된 복잡한 스키마보다 단순하고 명확한 예시가 더 효과적
- **디버깅**: 문제 발생 시 자동 생성 코드보다 직접 작성 코드가 디버깅하기 쉬움

### 6.3. 모델 동작 이해의 중요성
- **파인튜닝 데이터 분석**: 모델이 어떤 패턴으로 학습했는지 이해 필수
- **프롬프트 vs 학습**: 모델이 프롬프트와 학습 패턴 중 무엇을 우선하는지 파악
- **디버깅 접근법**: 기술적 문제뿐만 아니라 모델 학습 패턴도 고려해야 함

## 7. 현재 상태
- `base-info`와 `free-text` 모두 올바른 JSON 구조로 응답 생성
- 줄바꿈(`\n`)이 SSE 스트리밍에서 정상적으로 표시
- JSON 파싱 실패 시에도 안전한 fallback 제공
- 프롬프트가 파인튜닝 패턴과 일치하여 LLM 응답 품질 향상
- 모든 출력이 한글 100%로 일관성 확보
- escaped_format 제거로 코드 복잡성 감소 및 디버깅 용이성 향상

---

## 8. 파인튜닝 모델 토크나이저 문제 해결 (2025-07-27)

### 8.1. 문제 상황
- **증상**: 파인튜닝 모델 응답이 완전히 깨짐
- **예시 응답**: "다음과같은 Zero Waste챌린지를추천 dramabus99", "동기부^{+}", "바CHAR 매트랙"
- **카테고리 오류**: 모든 챌린지가 잘못된 카테고리로 분류
- **한글 토큰화 실패**: 의미불명한 문자열 생성

### 8.2. 원인 분석
#### 🔍 전체 상황 정리

**1. Mistral 모델의 기본 특성**
```
기본 Mistral-7B-Instruct는 [INST] 태그 형식으로 설계됨:
<s>[INST] 사용자 질문 [/INST] 모델 응답 </s>
```

**2. 우리 파인튜닝 데이터 형식 (v3, v4)**
```json
{
  "instruction": "너는 피드백 어시스턴트야...",
  "input": "JSON 입력",
  "output": "한글 응답"
}
```
**→ 순수 텍스트, [INST] 태그 없음**

**3. 문제 발생 지점**
```
파인튜닝 학습 시:
- 입력: "너는 친환경 챌린지 추천 챗봇이야..."
- 출력: "{"recommend": "...", "challenges": [...]}"

실제 vLLM 사용 시 (chat_template 적용):
- 입력: "<s>[INST] 너는 친환경 챌린지 추천 챗봇이야... [/INST]"
- 모델: "??? 이게 뭔 형식이지? 학습 때 본 적 없는데?"
```

**4. 해결책: chat_template 단순화**
```python
# 기존 (복잡한 Mistral Instruct 템플릿)
"chat_template": "{% for message in messages %}{% if message['role'] == 'user' %}<s>[INST] {{ message['content'] }} [/INST]..."

# 수정 후 (단순화)
"chat_template": "{{ messages[0].content }}"
```

#### 상세 분석
**Before (문제 상황):**
```
사용자 → "안녕하세요"
vLLM → "<s>[INST] 안녕하세요 [/INST]" (자동 변환)
파인튜닝 모델 → "이상한 형식이네... 학습 때 안 봤는데?" → 깨진 응답
```

**After (해결 후):**
```
사용자 → "안녕하세요"
vLLM → "안녕하세요" (그대로 전달)
파인튜닝 모델 → "아! 이 형식 알아! 학습 때 봤어!" → 정상 응답
```

#### 핵심 포인트
1. **파인튜닝 데이터**: [INST] 태그 **없이** 학습
2. **Mistral 기본**: [INST] 태그 **있어야** 정상 작동
3. **우리 모델**: 파인튜닝으로 [INST] 태그 **없는** 형식에 특화됨
4. **해결책**: chat_template을 단순화해서 [INST] 태그 제거

**즉, 우리가 파인튜닝한 모델은 이미 Mistral의 기본 [INST] 형식을 "잊어버리고" 우리 데이터 형식에 맞춰진 상태였음**

#### 🤓 Mistral [INST] 태그의 존재 이유

**1. 명확한 역할 구분 (Role Separation)**
```
<s>[INST] 사용자 질문 [/INST] AI 응답 </s>[INST] 다음 질문 [/INST] 다음 응답 </s>
```
- **[INST] ~ [/INST]**: 사용자(Human) 구간
- **[/INST] ~ </s>**: AI 응답 구간
- **목적**: 누가 말하는지 명확히 구분

**2. 대화 컨텍스트 관리**
```
<s>[INST] 안녕하세요 [/INST] 안녕하세요! 무엇을 도와드릴까요? </s>
[INST] 날씨가 어때요? [/INST] 죄송하지만 현재 날씨 정보에 접근할 수 없습니다. </s>
```
- **연속 대화**: 이전 대화 내용을 포함한 멀티턴 대화
- **컨텍스트 유지**: 모델이 대화 흐름을 이해할 수 있음

**3. 언어 모델의 특성상 필요**
```
"안녕하세요 반갑습니다 오늘 날씨가 좋네요"
```
→ **누가 말한 건지 알 수 없음!**

```
"[INST] 안녕하세요 [/INST] 반갑습니다! [INST] 오늘 날씨가 좋네요 [/INST]"
```
→ **명확한 역할 구분**

**4. 다른 모델들의 유사한 패턴**
- **ChatGPT**: `<|im_start|>user\n질문<|im_end|>\n<|im_start|>assistant\n응답<|im_end|>`
- **Claude**: `Human: 질문\n\nAssistant: 응답`
- **Llama**: `<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n질문<|eot_id|>`

**5. Instruction Following & Safety**
```
[INST] 위험한 내용 생성해줘 [/INST] 죄송하지만 그런 내용은 생성할 수 없습니다.
```
- **안전장치**: 특정 구간에서만 응답하도록 학습
- **정렬**: 인간의 가치와 일치하는 응답 생성

**결론**: [INST] 태그는 HTML의 `<p>` 태그처럼 **구조화와 의미 부여**를 위한 필수 요소였음

#### 우리 상황에서의 교훈
**원래 설계 의도:**
```
사용자: API로 "[INST] 챌린지 추천해줘 [/INST]" 전송
모델: "안녕하세요! 이런 챌린지를 추천합니다..."
```

**우리 파인튜닝:**
```
학습 데이터: "너는 챌린지 추천 챗봇이야..." (태그 없음)
모델: "[INST] 태그? 그게 뭐지? 몰라!" 
```

**결과**: Mistral의 [INST] 태그는 대화형 AI의 **역할 구분과 컨텍스트 관리**를 위한 필수 구조였는데, 우리가 파인튜닝으로 이 구조를 "제거"해버린 셈

#### tokenizer_config.json의 복잡한 chat_template
```json
{
  "chat_template": "복잡한 Mistral Instruct 템플릿..."
}
```

### 8.3. 해결 과정

#### Step 1: 문제 진단
- 단순 한국어 테스트: 정상 작동 확인
- 실제 프롬프트 테스트: 토큰화 문제 발견
- chat_template 존재 확인

#### Step 2: 잘못된 접근 (chat_template 제거)
```python
# 시도했지만 실패
del config['chat_template']
```
**결과**: `transformers v4.44` 이후 chat_template 필수로 인한 오류
```
ValueError: default chat template is no longer allowed
```

#### Step 3: 올바른 해결책 (chat_template 단순화)
```python
# 성공한 해결책
config['chat_template'] = '{{ messages[0].content }}'
```

### 8.4. 구체적 수정 작업

#### 파일 위치
```
/home/wonwonfll/mistral_fintuned4/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/0d4b76e1efeb5eb6f6b5e757c79870472e04bd3a/tokenizer_config.json
```

#### 수정 명령어
```bash
# 백업 생성
cp tokenizer_config.json tokenizer_config.json.backup

# Python으로 수정
python3 -c "
import json
with open('tokenizer_config.json.backup', 'r') as f:
    config = json.load(f)
config['chat_template'] = '{{ messages[0].content }}'
with open('tokenizer_config.json', 'w') as f:
    json.dump(config, f, indent=2)
"
```

#### vLLM 서버 재시작
```bash
./stop_services.sh
python3 -m vllm.entrypoints.openai.api_server \
  --model /path/to/model \
  --host 0.0.0.0 \
  --port 8800 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 20000 \
  --max-num-seqs 64
```

### 8.5. 핵심 인사이트

#### ResponseSchema vs 실제 사용 패턴
- **발견**: ResponseSchema가 설정되어 있지만 실제로는 fallback 로직이 대부분 실행됨
- **원인**: 파인튜닝 데이터와 LangChain 자동 생성 스키마 불일치
- **결론**: 파인튜닝 모델에서는 단순한 프롬프트가 더 효과적

#### Mistral 모델 특성과 파인튜닝
- **기본 Mistral**: [INST] 태그 사용 설계
- **파인튜닝 후**: 학습 데이터 패턴에 따라 달라짐
- **교훈**: 파인튜닝 시 사용한 데이터 형식과 추론 시 형식이 일치해야 함

#### chat_template의 중요성
- **transformers v4.44+**: chat_template 필수
- **해결책**: 제거가 아닌 단순화
- **최종 형태**: `{{ messages[0].content }}` (순수 프롬프트 전달)

### 8.6. 예상 결과
수정 후 예상되는 개선 사항:
1. **토큰화 정상화**: 한국어 텍스트 정상 처리
2. **JSON 구조 안정화**: 올바른 recommend/challenges 구조
3. **카테고리 정확성**: 올바른 카테고리 분류
4. **응답 품질 향상**: 파인튜닝 학습 패턴과 일치

### 8.7. 향후 고려사항

#### 파인튜닝 데이터 개선
- v4 데이터를 [INST] 형식으로 재생성 고려
- 일관된 템플릿 사용으로 일반화 성능 향상

#### 모니터링 포인트
- 기본 Mistral 능력 유지 여부 확인
- 다양한 프롬프트 패턴에 대한 강건성 테스트
- 파인튜닝 성능과 일반화 성능 간 균형

---
# 2025-07-28 챗봇 카테고리 변경 로직 구현 및 수정

## 1. 문제 상황
- **증상**: free-text 챗봇에서 "플로깅 관련 챌린지 추천해주세요" 요청 시 카테고리가 변경되지 않음
- **결과**: "플로깅" 키워드가 있음에도 불구하고 `"category": "ZERO_WASTE"`로 응답
- **예상**: `"category": "PLOGGING"`으로 변경되어야 함

## 2. 문제 분석 과정

### 2.1. 초기 가설 - process_chat 함수 문제
- **가설**: `LLM_chatbot_free_text_model.py`의 `process_chat` 함수에서 카테고리 변경 로직 부재
- **조치**: `process_chat` 함수에 키워드 기반 카테고리 변경 로직 추가
- **결과**: 여전히 작동하지 않음

### 2.2. 실제 문제 발견 - 라우터에서 process_chat 미호출
- **핵심 문제**: free-text 엔드포인트가 `process_chat` 함수를 호출하지 않음
- **실제 동작**: 라우터(`chatbot_router.py`)에서 직접 처리
- **증거**: 디버깅 로그 `🚀🚀🚀 FREE-TEXT PROCESS CHAT START 🚀🚀🚀`가 전혀 출력되지 않음

### 2.3. 변수명 충돌 문제
- **최종 문제**: 라우터에서 지역 변수 `category_keywords`가 import한 글로벌 변수를 덮어씀
- **충돌 코드**:
```python
# 라우터 내부 (지역 변수)
category_keywords = ["원래", "처음", "이전", "원래대로", "기존", "카테고리"]

# constants.py에서 import한 글로벌 변수 (덮어써짐)
from ..model.chatbot.chatbot_constants import category_keywords  # dict 타입
```

## 3. 해결 과정

### 3.1. 카테고리별 키워드 매핑 정의
`chatbot_constants.py`에 카테고리별 연관 키워드 추가:

```python
# 카테고리별 연관 키워드 매핑
category_keywords = {
    "제로웨이스트": ["제로웨이스트", "일회용", "플라스틱", "텀블러", "분리수거", "재활용", "쓰레기", "포장재"],
    "플로깅": ["플로깅", "운동", "조깅", "러닝", "달리기", "걷기", "산책", "운동하면서", "뛰면서", "스포츠"],
    "탄소발자국": ["탄소발자국", "탄소", "탄소중립", "온실가스", "기후변화", "대중교통", "자전거", "도보"],
    "에너지 절약": ["에너지", "절약", "전기", "전력", "전등", "조명", "콘센트", "에어컨", "난방"],
    "업사이클": ["업사이클", "재활용", "새활용", "DIY", "만들기", "창작", "재사용", "변형"],
    "문화 공유": ["문화", "공유", "소셜", "SNS", "캠페인", "홍보", "공유하기", "알리기", "전파"],
    "디지털 탄소": ["디지털", "온라인", "인터넷", "스마트폰", "컴퓨터", "전자기기", "클라우드", "데이터"],
    "비건": ["비건", "채식", "식물성", "동물", "고기", "유제품", "채소", "과일", "식단"]
}
```

### 9.3.2. 라우터에 카테고리 변경 로직 추가
`chatbot_router.py`의 free-text 엔드포인트에 키워드 기반 카테고리 변경 로직 구현:

```python
# 1. 카테고리 변경 로직 (키워드 기반)
current_category = conversation_states[sessionId].get("category", "제로웨이스트")
message_lower = message.lower()

# 카테고리 변경 검사
category_changed = False
for category, keywords in category_keywords.items():
    if any(keyword in message_lower for keyword in keywords):
        conversation_states[sessionId]["category"] = category
        current_category = category
        category_changed = True
        break
```

### 3.3. 변수명 충돌 해결
라우터의 지역 변수명을 변경하여 충돌 방지:

```python
# 기존 (충돌 발생)
category_keywords = ["원래", "처음", "이전", "원래대로", "기존", "카테고리"]

# 수정 (충돌 해결)
category_reset_keywords = ["원래", "처음", "이전", "원래대로", "기존", "카테고리"]
```

### 3.4. 프롬프트 출력 형식 개선
title에 번호를 붙이도록 프롬프트 수정:

```python
# 기존
"title": "제로웨이스트 실천"

# 수정
"title": "1. 제로웨이스트 실천"
```

## 4. 수정된 파일 목록

### 4.1. `Text/LLM/model/chatbot/chatbot_constants.py`
- **추가**: `category_keywords` dict 정의
- **목적**: 카테고리별 연관 키워드 매핑

### 4.2. `Text/LLM/router/chatbot_router.py`
- **추가**: `category_keywords` import
- **추가**: 카테고리 변경 로직 구현
- **수정**: 지역 변수명 `category_reset_keywords`로 변경

### 4.3. `Text/LLM/model/chatbot/LLM_chatbot_free_text_model.py`
- **추가**: 디버깅 로그 (실제로는 사용되지 않음)
- **추가**: base_info_category 기본값 설정 로직
- **수정**: 프롬프트 출력 예시에 번호 추가

### 4.4. `Text/LLM/model/chatbot/LLM_chatbot_base_info_model.py`
- **수정**: 프롬프트 출력 예시에 번호 추가

## 5. 최종 결과
- **성공**: "플로깅 관련 챌린지 추천해주세요" → `"category": "PLOGGING"`
- **성공**: title에 번호 추가 → `"title": "1. 플로깅 시작하기"`
- **성공**: 키워드 기반 실시간 카테고리 변경 가능

##.6. 아키텍처 이해

### 6.1. 실제 동작 흐름
1. **클라이언트 요청** → `chatbot_router.py`
2. **라우터에서 직접 처리** (process_chat 함수 미사용)
3. **카테고리 변경 로직** → 라우터 내부에서 실행
4. **RAG 검색** → 변경된 카테고리로 문서 검색
5. **LLM 응답 생성** → 해당 카테고리 컨텍스트로 응답

### 6.2. free-text의 독립성
- **base-info 불필요**: free-text는 단독 실행 가능
- **기본 카테고리**: "제로웨이스트"로 시작
- **동적 변경**: 사용자 메시지 키워드에 따라 실시간 카테고리 변경

# 2025-07-29 챗봇 temperature 지정

### 1. 현재 temperature 인자가 설정되지 않아 기본값(보통 1.0)이 사용되고 있었음.

```py
# free-text 모델
payload = {
    "model": "...",
    "messages": [{"role": "user", "content": prompt}],
    "stream": True,
    "max_tokens": 512
    # temperature 없음 → 기본값 1.0 사용
}

# base-info 모델  
payload = {
    "model": "...",
    "messages": [{"role": "user", "content": prompt}],
    "stream": True,
    "max_tokens": 2048
    # temperature 없음 → 기본값 1.0 사용
}
```

### 1.1.파인튜닝과 Temperature의 관계

1.1.1. 파인튜닝은 모델의 "지식"을 바꿈
- 특정 도메인(친환경 챌린지)에 대한 이해력 향상
- 출력 형식과 스타일 학습
1.1.2.Temperature는 추론 시의 "창의성"을 조절
- 파인튜닝된 모델도 추론 시 temperature 설정 필요
- 낮은 temperature: 일관된, 예측 가능한 응답
- 높은 temperature: 다양한, 창의적인 응답

### 1.2. 각 엔드포인트에 맞게 temperature 설정 추가
```py
    "max_tokens": 512,
    "temperature": 0.7,
    "do_sample": True
```

### 1.3. 설정한 Temperature 값의 의미
1.3.1. free-text 모델: temperature=0.7
- 자유로운 대화형 챌린지 추천
- 적당한 창의성과 일관성의 균형
- 다양한 표현과 스타일 허용
1.3.2. base-info 모델: temperature=0.5
- 구조화된 챌린지 추천
- 더 일관된 형식과 내용
- 예측 가능한 JSON 구조 유지


### 추가 개선 사항
#### do_sample=True도 추가함
- do_sample=False: 항상 가장 확률이 높은 토큰 선택 (temperature 무시)
- do_sample=True: temperature에 따라 확률적 샘플링