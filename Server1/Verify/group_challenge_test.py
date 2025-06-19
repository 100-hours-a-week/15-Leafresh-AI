# test_prompt_storage.py
from model.group_prompt_generator import get_or_create_group_prompt
import ssl

challenge_id = 9999
challenge_name = "퇴근 후 텀블러 사용하기"
challenge_info = "퇴근길에 일회용 컵 대신 텀블러를 사용하는 행동을 유도하는 챌린지입니다."

prompt = get_or_create_group_prompt(challenge_id, challenge_name, challenge_info)
print(prompt)
print(ssl.OPENSSL_VERSION)