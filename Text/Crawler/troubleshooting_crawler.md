**날짜:** 25-07-10

## 최근 변경사항

### 1. FastAPI 통합
- **변경사항:** 크롤러가 `Text/LLM/main.py`의 FastAPI 서버에 통합됨
- **엔드포인트:** `/ai/crawler/run` (POST) - 크롤러 수동 실행
- **위치:** `Text/LLM/router/crawler_router.py`

### 2. 백그라운드 스레딩
- **변경사항:** main.py 시작 시 크롤러가 백그라운드 스레드에서 실행
- **동작:** FastAPI 서버가 즉시 시작되고, 크롤러가 동시에 실행됨
- **코드 위치:** `Text/LLM/main.py` - `threading.Thread(target=generate_challenge_docs, daemon=True).start()`

### 3. 서버 시작 플로우
- **이전:** 크롤러가 끝날 때까지 기다린 후 서버 시작
- **현재:** 서버가 즉시 시작되고, 크롤러는 백그라운드에서 실행
- **장점:** 서버 시작 지연 없음, API 엔드포인트 즉시 사용 가능

### 4. [25-07-10] Crawler/Embed 파이프라인 구조 분리 및 개선
- **변경사항:**
    - `generate_challenge_docs.py`는 백그라운드 스레드에서 싱행되어 챌린지 데이터(고정+크롤링)만 생성-> `challenge_docs.txt`에 저장 (임베딩 X)
    - `embed_init.py`는 오직 `challenge_docs.txt` 파일을 읽어서 임베딩 및 Qdrant 저장만 수행 (크롤링/데이터 생성 X)
    - `embed_init.py`에서 크롤링/데이터 생성 관련 코드 완전 제거
    - 파일 기반 파이프라인으로 역할이 명확히 분리되어 유지보수, 자동화, 테스트 용이
    - 중복 실행/무한루프 위험 해소
    - 


## 파일 구조
```
Text/
├── Crawler/
│   ├── generate_challenge_docs.py  # 메인 크롤러 로직
│   ├── embed_init.py              # 임베딩 및 Qdrant 저장
│   └── challenge_docs.txt         # 생성된 챌린지 데이터 (gitignored)
└── LLM/
    ├── main.py                    # 크롤러가 통합된 FastAPI 서버
    └── router/
        └── crawler_router.py      # 크롤러 API 엔드포인트
```

## API 엔드포인트
- 서버 시작 시 크롤러가 자동으로 실행됨 (백그라운드 스레드)
- `POST /ai/crawler/run` - 크롤러 수동 실행 가능 
```bash
curl -X POST http://localhost:8000/ai/crawler/run
```

## 의존성
- FastAPI
- threading (내장)
- requests, BeautifulSoup (크롤링용)
- SentenceTransformer, Qdrant (임베딩용)

## 참고사항
- main.py 시작 시 크롤러가 자동으로 실행됨
- challenge_docs.txt는 생성된 데이터를 커밋하지 않도록 gitignored됨
- 백그라운드 스레딩으로 서버가 즉시 시작되면서 크롤러가 동시에 실행됨
- 추후에 수동으로 크롤러 실행 하고 싶으면 `curl -X POST http://localhost:8000/ai/crawler/run` 입력