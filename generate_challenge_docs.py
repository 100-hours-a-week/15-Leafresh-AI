# generate_challenge_docs_unified.py
import random

def generate_challenge_docs(file_path="challenge_docs.txt", mode="random", num_paragraphs=100):# fixed, random으로 변경 가능
    """
    챌린지 문서를 생성합니다.

    Args:
        file_path (str): 저장할 파일 경로
        mode (str): "fixed" (고정 데이터) 또는 "random" (랜덤 조합 데이터)
        num_paragraphs (int): 랜덤 모드일 때 생성할 문단 수 (default: 100)
    """

    # 고정 데이터
    fixed_contents = [
        "플라스틱 사용을 줄이기 위해 텀블러를 사용하고, 장바구니를 지참합시다.",
        "제로웨이스트 생활을 실천하기 위해 쓰레기를 줄이고 재사용 가능한 제품을 사용합시다.",
        "탄소중립을 위해 대중교통을 이용하고, 걷기와 자전거 타기를 생활화합시다.",
        "에너지를 절약하기 위해 불필요한 전기 제품을 끄고, 에너지 효율이 높은 제품을 사용합시다.",
        "비건 식단을 시도하여 축산업에 의한 탄소 배출을 줄입시다."
    ]

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
        "일회용품 대신 다회용품을 사용하자."
    ]

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            if mode == "fixed":
                for line in fixed_contents:
                    f.write(line + "\n\n")
                print(f"✅ 고정 모드: {len(fixed_contents)}개 문단 저장 완료")

            elif mode == "random":
                for _ in range(num_paragraphs):
                    paragraph = " ".join(random.sample(random_sentences, k=3))  # 문장 3개 랜덤 조합
                    f.write(paragraph + "\n\n")
                print(f"✅ 랜덤 모드: {num_paragraphs}개 문단 저장 완료")

            else:
                print(f"❌ 오류: 지원하지 않는 모드입니다. ('fixed' 또는 'random' 사용)")

    except Exception as e:
        print(f"❌ 파일 생성 실패: {str(e)}")

# 테스트용 실행 코드
if __name__ == "__main__":
    # 예시: "fixed" 또는 "random" 모드 선택
    generate_challenge_docs("challenge_docs.txt", mode="random")
    # generate_challenge_docs("challenge_docs.txt", mode="random", num_paragraphs=100)