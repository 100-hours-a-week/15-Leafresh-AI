# AI 기반의 지속가능한 친환경 챌린지 및 커뮤니티 플랫폼

### 🌿 ***'Small Tasks, Global Ripple'*** 

> "당신의 행동이 지구에게 어떤 영향을 줄까요?"
> 일상 속 작은 행동 하나하나가 모여 누구나 쉽고 재밌게 지속가능한 삶을 실천할 수 있도록 도움 제공

### [서비스 바로가기](https://leafresh.app/)

<br>

## 📌 Quick View

### [영상 바로가기](https://drive.google.com/file/d/1O8r-uZpLbOSZO7-Ohy88Csapj6l6PVkOI/view?usp=sharing)

<img src="https://github.com/user-attachments/assets/a1bfcdfe-091a-4a44-b003-3383dcb38d1f" width="750">

<br>
<br>

## ⚒️ Usage Stack

분류 | 사용 기술
-- | --
AI Model | `Vertex AI (Gemini-2.0-flash) API`, `LLaVA-13B`, `Mistral-7B`
Server | `Python`, `FastAPI`, `Cloud Pub/Sub`, `GCS`, `SSE`, `MongoDB`
LLM Orchestration | `LangChain`, `RAG`, `VectorDB (QdrantDB)`

<br>

## 👉🏻 Role & Responsibilities

no. | 기능 | 모델명 | 설명 | 사용 모델 
-- | -- | -- | -- | --
1 | 챌린지 이미지 인증 | verify | 유저 인증 이미지를 기반으로 AI가 자동 검증 | API -> `LLaVA-13B`
2 | 챌린지 생성 검열 | censorship | 챌린지 생성 시 AI를 통해 중복/부적절 항목 필터링 | API -> `Mistral-7B`
3 | 챌린지 추천 챗봇 | chatbot | 개인 취향 기반 챌린지 추천 | API -> `Mistral-7B`
4 | 주간 피드백 생성 | feedback | 주간 챌린지 활동을 분석하여 요약 피드백 제공 | API -> `Mistral-7B`

<br>

## 📈 Model Performance

Model | Version | Accuracy | 개선 사항
-- | -- | -- | --
Censorship Model | v1.1 -> v1.2 | 66.00% -> `96.00%` | Rule-based 필터링 추가, 프롬프트 개선
Verify Model | v1.1 -> v1.2 | 75.71% -> `98.68%` | LangChain 적용, 이미지 리사이징, 프롬프트 개선

<br>

## 👉🏻 FastAPI end-point

[AI API 설계 보고서](https://github.com/100-hours-a-week/15-Leafresh-wiki/wiki/AI-%EB%AA%A8%EB%8D%B8-API-%EC%84%A4%EA%B3%84)

no. | Note | Mothod | Endpoint | Role
-- | -- | -- | -- | --
1 | 사진 인증 요청 <br> : BE -> AI | POST | /ai/image/verification | 이미지 인증 요청 전송 (이미지 포함)
2 | 인증 결과 <br> : AI -> BE | POST | /api/verifications/{verificationId}/result | AI의 인증 결과 콜백 수신 <br> (모델 추론 결과 반환)
3 | 카테고리 기반 추천 <br> : BE -> AI | POST | /ai/chatbot/recommendation/base-info | 선택 기반 챌린지 추천 챗봇
4 | 자유 입력 추천 <br> : BE -> AI | POST | /ai/chatbot/recommendation/free-text | 자연어 기반 챌린지 추천 챗봇
5 | 생성 검열 요청 <br> : BE -> AI | POST | /ai/challenges/group/validation | 챌린지 생성 요청 시, <br> 제목 유사성과 중복 여부를 기반으로 생성 가능성 판단
6 | 주간 피드백 생성 요청 <br> : BE -> AI | POST | /ai/feedback | 사용자가 마이페이지에서 요청시, <br> 사용자 주간 데이터를 기반으로 피드백 생성
7 | 피드백 결과 <br> : AI -> BE | POST | /api/members/feedback/result | 피드백 결과 콜백 수신 
8 | (추가) 서버 헬스 체크 | GET | /health | 서버 실행 여부 판단

<br>

## 👉🏻 Service Architecture

### MVP - Bigbang Architecture

<img width="1000" alt="MVP_빅뱅배포_아키텍처" src="https://github.com/user-attachments/assets/4e81c354-1616-44a0-bd41-f0df8b61b4d8" />

<br>

### v2 - CI/CD Architecture

<img width="1000" alt="Leafresh_AI_아키텍처" src="https://github.com/user-attachments/assets/3979c999-945f-43f9-acdc-20739b0500d7" />

<br>
<br>







