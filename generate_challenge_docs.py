# generate_challenge_docs_unified.py
# 챌린지 문서 생성기
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
    "[제로웨이스트] 텀블러 사용과 장바구니 지참으로 플라스틱 사용을 줄입시다.",
    "[제로웨이스트] 비닐봉지 대신 에코백을 사용하여 쓰레기를 줄여보세요.",
    "[플로깅] 출퇴근 또는 산책 시 플로깅을 실천해 보세요.",
    "[플로깅] 주말마다 공원에서 플로깅 캠페인을 열어보세요.",
    "[비건 식당] 1주일에 하루 비건 식당 방문으로 환경 보호에 동참하세요.",
    "[비건 식당] 사내 식당에서 채식 메뉴를 선택하는 것도 좋은 방법입니다.",
    "[에너지 절약] 퇴근 시 전자기기 전원을 완전히 꺼주세요.",
    "[에너지 절약] 밝은 낮에는 조명을 끄고 자연광을 활용하세요.",
    "[중고거래] 쓰지 않는 물건을 중고 거래 플랫폼에 등록해보세요.",
    "[중고거래] 사무용품도 중고로 구매하여 자원을 절약해보세요.",
    "[탄소발자국] 출퇴근 시 대중교통이나 자전거를 이용해보세요.",
    "[탄소발자국] 자신의 탄소발자국을 계산해보고 감축 계획을 세워보세요.",
    "[서적/영화] 환경 관련 다큐멘터리를 시청하고 팀원과 토론을 나눠보세요.",
    "[서적/영화] 제로웨이스트 실천 서적을 읽고 사내 독서 모임을 가져보세요.",
    "[디지털 탄소] 불필요한 이메일 정리와 클라우드 저장소 최적화를 해보세요.",
    "[디지털 탄소] 업무 종료 시 PC 절전 모드를 설정하여 전력 소비를 줄여보세요."
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