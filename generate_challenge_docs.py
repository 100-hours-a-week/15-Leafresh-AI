# generate_challenge_docs_unified.py
# 챌린지 문서 생성기
import random

def generate_challenge_docs(file_path="challenge_docs.txt", mode="random", num_paragraphs=100):
    """
    챌린지 문서를 생성합니다.

    Args:
        file_path (str): 저장할 파일 경로
        mode (str): "fixed" (고정 데이터) 또는 "random" (랜덤 조합 데이터)
        num_paragraphs (int): 랜덤 모드일 때 생성할 문단 수 (default: 100)
    """

    # 랜덤 데이터 후보 문장
    random_sentences = [
    "텀블러 사용으로 플라스틱 쓰레기를 줄이자.",
    "비건 식단을 통해 탄소 배출을 줄이자.",
    "에코백을 들고 다니며 비닐봉지 사용을 줄이자.",
    "대중교통을 이용해 탄소중립 실천하기.",
    "재활용 가능한 제품을 구매하자.",
    "제로웨이스트를 목표로 생활하자.",
    "불필요한 전등 끄기로 에너지 절약하자.",
    "걷기와 자전거 타기를 생활화하자.",
    "물 절약을 위해 양치컵 사용을 생활화하자.",
    "일회용품 대신 다회용품을 사용하자.",
    "사무실에서도 텀블러 사용을 생활화하자.",
    "분리수거를 정확하게 실천하자.",
    "환경 관련 책을 읽고 사내 독서 토론회를 열자.",
    "불필요한 이메일은 삭제해 디지털 탄소를 줄이자.",
    "점심시간에 채식 위주의 메뉴를 선택하자.",
    "재활용이 가능한 커피캡슐을 사용하자.",
    "온라인 회의 시 배경 조명 줄이기 실천하자.",
    "재사용 가능한 메모지를 사용하자.",
    "대체육 제품을 주 1회 이상 소비해보자.",
    "전력 소비가 적은 가전제품을 선택하자.",
    "지구의 날에는 전등을 끄고 생활해보자.",
    "사무실 내부 온도를 적정하게 유지하자.",
    "택시 대신 대중교통 앱을 활용하자.",
    "환경 관련 영상을 시청하고 팀과 공유하자.",
    "플라스틱 빨대 대신 종이 빨대 사용을 생활화하자.",
    "이메일 대신 구두 전달로 정보 공유해보자.",
    "프린트는 흑백으로, 필요할 때만 출력하자.",
    "중고 물품 나눔 플랫폼을 사내에 소개하자.",
    "직장 내 자전거 거치대를 활용해보자.",
    "직장 동료와 함께 분기마다 친환경 캠페인을 기획하자."
    ]
 

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            if mode == "random":
                for _ in range(num_paragraphs):
                    num_sentences = random.randint(2, 5)
                    paragraph = " ".join(random.sample(random_sentences, num_sentences))
                    f.write(f"{paragraph}\n\n")
                print(f"랜덤 모드: {num_paragraphs}개 문단 저장 완료")
            else:
                print(f"오류: 지원하지 않는 모드입니다. ('fixed' 또는 'random' 사용)")

    except Exception as e:
        print(f"파일 생성 실패: {str(e)}")