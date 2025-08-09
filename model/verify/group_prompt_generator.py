import google.generativeai as genai
from model.verify.mongodb import insert_prompt, get_prompt_by_id, prompt_exists

def generate_group_prompt(challenge_name: str, challenge_info: str) -> str:      
    model = genai.GenerativeModel("gemini-2.0-flash")

    system_prompt_message = (
        "당신은 사용자 챌린지를 위한 프롬프트를 설계하는 AI입니다. "
        "입력받은 [챌린지 이름]과 [챌린지 설명]을 기반으로 이미지 검증에 사용할 수 있는 프롬프트를 아래와 같은 형식으로 생성해주세요.\n"

        "--- 예시 ---\n"
        "이 이미지는 '안쓰는 콘센트 뽑기' 챌린지에 적합한가요? \n\n"

        "※ 조건: 콘센트(전기 소켓)에서 전기 플러그가 실제로 뽑힌 장면이 명확하게 보여야 합니다.\n"
        "-> 랜선, HDMI 케이블, 모니터 코드 등을 뽑은 것은 적합하지 않습니다.\n\n"

        "흐리거나 블러 처리된 이미지는 '아니오'입니다.\n"
        "예/아니오로만 대답해주세요.\n"
        "-------------\n"

        "결과는 반드시 사용자의 이미지가 챌린지에 적합한지 판별할 수 있는 형태여야 하며, 예/아니오로만 대답하게 유도해야 합니다.\n"
    )

    prompt_input = f"[챌린지 이름]: {challenge_name} \n[챌린지 설명] : {challenge_info}"

    response = model.generate_content(
        [system_prompt_message, prompt_input],       # Chat 기반의 멀티턴 메시지 구조를 지원하므로 두개로 나누어서 함 
        generation_config={
            "temperature": 0.4,
            "top_p": 1,
            "top_k": 32,
            "max_output_tokens": 512
        }
    )

    return response.text
    
# 프롬프트가 없으면 생성 후 저장, 있으면 불러오기
def get_or_create_group_prompt(challenge_id: int, challenge_name: str, challenge_info: str) -> str:
    if prompt_exists(challenge_id):
        return get_prompt_by_id(challenge_id)
    
    prompt_text = generate_group_prompt(challenge_name, challenge_info)
    insert_prompt(challenge_id, challenge_name, prompt_text)

    return prompt_text
