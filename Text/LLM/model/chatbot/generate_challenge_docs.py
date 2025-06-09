# generate_challenge_docs.py
# 챌린지 문서 생성기
import random
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime
from urllib.parse import urljoin, urlparse

def clean_text(text):
    """텍스트 정제 함수"""
    text = ' '.join(text.split())
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = ' '.join(text.split())
    return text

def is_valid_url(url, base_domain):
    """URL이 같은 도메인인지 확인"""
    try:
        return urlparse(url).netloc == base_domain
    except:
        return False

def fetch_environmental_news():
    """환경 관련 뉴스를 크롤링"""
    news_items = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    # 환경일보 크롤링
    try:
        base_url = "https://www.hkbs.co.kr"
        visited_list_urls = set() # 방문한 목록 페이지 URL
        urls_to_visit = [f"{base_url}/news/articleList.html?sc_section_code=S1N3"]
        article_links = set() # 수집한 기사 상세 페이지 링크
        
        while urls_to_visit and len(visited_list_urls) < 10:  # 최대 20개 목록 페이지까지
            current_list_url = urls_to_visit.pop(0)
            if current_list_url in visited_list_urls:
                continue
                
            print(f"환경일보 목록 크롤링 중: {current_list_url}")
            response = requests.get(current_list_url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 변경: h4.titles 안에 있는 a 태그 선택
            article_elements = soup.select("h4.titles a") 
            found_article_links_on_page = 0
            for link_element in article_elements: # 링크 요소 순회
                href = link_element.get('href')
                article_url = urljoin(base_url, href)
                
                # 'articleView.html?idxno=' 패턴을 포함하고, 동일 도메인인지 확인 (필터링 유지)
                if "articleView.html?idxno=" in article_url and is_valid_url(article_url, "www.hkbs.co.kr"):
                    article_links.add(article_url)
                    found_article_links_on_page += 1
                    
            print(f"  -> 현재 목록 페이지에서 'h4.titles a'로 찾은 잠재적 기사 링크 수: {found_article_links_on_page}") # 로깅 메시지 수정

            # 다음 페이지 링크 수집 (기사 목록 링크만)
            next_links = soup.select("a[href*='articleList']")
            for link in next_links:
                next_url = urljoin(base_url, link['href'])
                if is_valid_url(next_url, "www.hkbs.co.kr") and next_url not in visited_list_urls:
                    urls_to_visit.append(next_url)
            
            visited_list_urls.add(current_list_url)
            time.sleep(0.3)  # 서버 부하 방지
            
    except Exception as e:
        print(f"환경일보 목록 크롤링 중 오류 발생: {str(e)}")

    print(f"총 환경일보 기사 상세 페이지 링크 수집 완료: {len(article_links)}")
    
    # 수집된 기사 링크 순회하며 상세 내용 크롤링
    for article_url in list(article_links):
        print(f"  기사 페이지 시도: {article_url}") # 시도하는 URL 로깅
        try:
            article_response = requests.get(article_url, headers=headers, timeout=10)
            article_soup = BeautifulSoup(article_response.text, 'html.parser')
            
            # 제목과 내용 추출 (실제 HTML 구조에 맞게 수정)
            title = article_soup.select_one("h3.titles, h2.titles, div.article-head-title, h3.heading")
            content = article_soup.select_one("div#article-view-content-div, div.article-view-content, div.article-body")
            
            print(f"    - 제목 요소 찾음: {title is not None}") # 제목 요소 찾았는지 로깅
            print(f"    - 본문 요소 찾음: {content is not None}") # 본문 요소 찾았는지 로깅

            if title and content:
                title_text = title.text.strip()
                content_text = content.text.strip()
                
                print(f"    - 제목 길이: {len(title_text) if title_text else 0}") # 제목 길이 로깅
                print(f"    - 본문 길이: {len(content_text) if content_text else 0}") # 본문 길이 로깅

                if title_text and content_text and len(title_text) > 10:
                    news_items.append({
                        "title": clean_text(title_text),
                        "content": clean_text(content_text),
                        "url": article_url
                    })
                    print(f"    - 기사 크롤링 성공 및 추가: {title_text[:30]}...") # 성공 시 로깅
                else:
                    print("    - 제목 또는 본문 내용이 짧거나 비어있음.") # 내용 부족 시 로깅
            else:
                print("    - 제목 또는 본문 요소를 찾지 못했습니다.") # 요소 못 찾음 시 로깅
            
            time.sleep(0.5)  # 서버 부하 방지
            
        except Exception as e:
            print(f"    - 기사 상세 페이지 크롤링 중 오류 발생: {str(e)}") # 오류 발생 시 로깅
            continue
            
    return news_items

def fetch_environmental_events():
    """환경 관련 행사 정보를 크롤링합니다."""
    events = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

def generate_challenge_from_news(news_item):
    """뉴스 기사를 참고하여 챌린지 문장 생성"""
    # 키워드 기반 카테고리 및 챌린지 매핑
    challenge_templates = {
        "플라스틱": {
            "category": "제로웨이스트",
            "templates": [
                "뉴스에서 본 플라스틱 문제를 해결하기 위해 {item} 사용을 실천하고 있습니다.",
                "플라스틱 사용을 줄이기 위해 {item}를 활용한 제로웨이스트 생활을 실천하고 있습니다.",
                "플라스틱 대신 {item}를 사용하여 환경 보호에 기여하고 있습니다."
            ],
            "items": ["텀블러", "장바구니", "다회용 용기", "비닐봉투 대신 장바구니", "일회용 컵 대신 머그컵"]
        },
        "에너지": {
            "category": "에너지절약",
            "templates": [
                "에너지 절약 뉴스를 보고 {item} 사용을 최적화하고 있습니다.",
                "에너지 소비를 줄이기 위해 {item}를 실천하고 있습니다.",
                "에너지 절약을 위해 {item}를 생활화하고 있습니다."
            ],
            "items": ["전등", "컴퓨터", "냉난방", "대중교통", "자전거"]
        },
        "비건": {
            "category": "비건",
            "templates": [
                "비건 식단 관련 뉴스를 보고 {item}를 실천하고 있습니다.",
                "비건 라이프스타일을 위해 {item}를 생활화하고 있습니다.",
                "비건 식단을 통해 {item}를 실천하고 있습니다."
            ],
            "items": ["채식 식단", "비건 레시피", "식물성 단백질", "지역 농산물"]
        },
        "플로깅": {
            "category": "플로깅",
            "templates": [
                "플로깅 관련 뉴스를 보고 {item}를 실천하고 있습니다.",
                "환경 정화를 위해 {item}를 생활화하고 있습니다.",
                "플로깅을 통해 {item}를 실천하고 있습니다."
            ],
            "items": ["주변 정화", "해변 정화", "공원 정화", "산책로 정화"]
        },
        "업사이클": {
            "category": "업사이클",
            "templates": [
                "업사이클링 뉴스를 보고 {item}를 실천하고 있습니다.",
                "자원 재활용을 위해 {item}를 생활화하고 있습니다.",
                "업사이클링을 통해 {item}를 실천하고 있습니다."
            ],
            "items": ["폐품 재활용", "중고거래", "재활용품 활용", "DIY 제작"]
        },
        "디지털": {
            "category": "디지털탄소",
            "templates": [
                "디지털 탄소 관련 뉴스를 보고 {item}를 실천하고 있습니다.",
                "디지털 탄소를 줄이기 위해 {item}를 생활화하고 있습니다.",
                "디지털 탄소 감소를 위해 {item}를 실천하고 있습니다."
            ],
            "items": ["이메일 정리", "클라우드 정리", "화면 밝기 조절", "불필요한 앱 삭제"]
        },
        "문화": {
            "category": "문화공유",
            "templates": [
                "환경 문화 관련 뉴스를 보고 {item}를 실천하고 있습니다.",
                "환경 문화 확산을 위해 {item}를 생활화하고 있습니다.",
                "환경 문화 공유를 위해 {item}를 실천하고 있습니다."
            ],
            "items": ["SNS 공유", "커뮤니티 활동", "워크숍 참여", "캠페인 참여"]
        }
    }
    
    # 뉴스 제목과 내용에서 키워드 매칭
    matched_category = None
    search_text = f"{news_item['title']} {news_item['content']}"
    
    # 일상 친환경 챌린지 관련 키워드 추가
    everyday_keywords = {
        "텀블러": "제로웨이스트",
        "장바구니": "제로웨이스트",
        "분리수거": "제로웨이스트",
        "절수": "에너지절약", # 물 절약도 에너지 절약과 연관
        "대중교통": "에너지절약", # 친환경 운송
        "새활용": "업사이클", # 업사이클과 유사
        "재활용": "업사이클",
        "채식": "비건",
        "플로깅": "플로깅",
        "줍깅": "플로깅",
        "전기차": "에너지절약", # 친환경 운송
        "폐기물": "제로웨이스트",
        "음식물 쓰레기": "제로웨이스트",
        "퇴비": "업사이클", # 음식물 쓰레기 퇴비화 등
        "미세먼지": "에너지절약", # 대중교통 이용 등과 연관
        "탄소중립": "에너지절약", # 폭넓은 개념
        "온실가스": "에너지절약"
    }

    # 일상 친환경 키워드 먼저 매칭 시도
    for keyword, category in everyday_keywords.items():
        if keyword in search_text:
            # 해당 키워드에 대한 템플릿/아이템이 challenge_templates에 정의되어 있는지 확인
            if category in [data["category"] for data in challenge_templates.values()]:
                 # 해당 카테고리의 템플릿/아이템 사용
                 for temp_keyword, data in challenge_templates.items():
                     if data["category"] == category:
                         matched_category = data
                         print(f"  - 키워드 매칭 성공: {keyword} -> 카테고리: {category}") # 키워드 매칭 로깅
                         break # 해당 카테고리를 찾으면 내부 루프 종료
            else:
                 # challenge_templates에 해당 카테고리가 없는 경우 (현재는 모든 카테고리 있음)
                 # 필요하다면 여기에 새로운 카테고리 및 템플릿 정의 로직 추가
                 pass # 현재는 기존 카테고리만 사용
            break # 첫 번째 매칭되는 일상 키워드 찾으면 루프 종료

    # 기존 키워드 매칭 시도 (일상 키워드에서 매칭되지 않은 경우)
    if not matched_category:
         for keyword, data in challenge_templates.items():
             if keyword in search_text:
                 matched_category = data
                 print(f"기존 키워드 매칭 성공: {keyword} -> 카테고리: {data['category']}") # 키워드 매칭 로깅
                 break

    if matched_category:
        template = random.choice(matched_category["templates"])
        item = random.choice(matched_category["items"])
        content = template.format(item=item)
        
        # 직종과 위치 랜덤 선택
        job_types = ["사무직", "영업직", "현장직", "재택근무"]
        locations = ["도시", "농촌", "산간", "해안"]
        
        return {
            "content": content,
            "metadata": {
                "category": matched_category["category"],
                "source": "뉴스기반",
                "location": random.choice(locations),
                "job_type": random.choice(job_types),
                "news_url": news_item["url"]
            }
        }
    return None

def generate_challenge_from_event(event_title):
    """행사 정보를 참고하여 챌린지 문장 생성"""
    # 행사 유형별 챌린지 템플릿
    # 현재는 사용하지 않으므로 빈 템플릿 반환
    return None

def generate_challenge_docs(file_path="challenge_docs.txt", mode="random", num_paragraphs=100):
    """
    Args:
        file_path (str): 저장할 파일 경로
        mode (str): "fixed" (고정 데이터) 또는 "random" (랜덤 조합 데이터)
        num_paragraphs (int): 랜덤 모드일 때 생성할 문단 수 (default: 100)
    """
    # 1. 기존 랜덤 문장들 (고정 데이터)
    random_sentences = [
        "도시에서 사무직으로 근무하며 점심시간에 제로웨이스트 도시락을 실천하고 있습니다.",
        "바다 근처에서 영업직으로 일하며 출퇴근 시 플로깅을 실천하고, 지역 커뮤니티에도 확산시키고 있습니다.",
        "산간 지역에서 재택근무하며 주 1회 가족과 함께 업사이클 공예를 진행하며 친환경 문화를 실천하고 있습니다.",
        "농촌에서 현장직 근무 중 비닐 대신 천 포대를 활용해 제로웨이스트 농작업을 실현하고 있습니다.",
        "도시에서 재택근무하며 디지털 탄소 절감을 위해 클라우드 정리와 화면 밝기 조절 캠페인을 진행하고 있습니다.",
        "현장직 근무자는 매일 차량 대신 자전거로 출퇴근하며 탄소발자국 줄이기에 참여하고 있습니다.",
        "영업직 근무 중 출장지에서 채식 식당을 탐방하며 비건 챌린지를 SNS에 공유하고 있습니다.",
        "사무실에서는 냉난방 온도를 2도 낮추는 에너지 절약 캠페인이 진행되고 있습니다.",
        "재택근무자는 회사 전체에 '이메일 청소의 날'을 제안하고 실행하여 디지털 탄소를 줄이고 있습니다.",
        "농촌 마을에서 플라스틱 병을 활용한 화분 만들기 워크숍을 통해 업사이클 문화를 확산시키고 있습니다.",
        "도시 재택근무 팀은 매월 '종이 없는 하루'를 지정해 문서 디지털화를 촉진하고 있습니다.",
        "산간 지역에서 플로깅 행사를 환경 교육과 결합하여 지역사회 참여를 높이고 있습니다.",
        "바닷가 인근 회사는 일회용품 제로 캠페인을 운영하며 인증 배지를 수여하고 있습니다.",
        "사무직 근무자는 비건 점심 도시락 릴레이를 조직해 동료들과 식단 변화를 나누고 있습니다.",
        "현장 근무 중 음료 캔을 자율 분리수거 코너에 모아 실천을 이어가고 있습니다.",
        "농촌 재택근무자는 매주 업사이클링 아이디어를 메신저에서 공유하고 실천을 유도하고 있습니다.",
        "영업직은 고객 방문 시 에코백을 활용해 불필요한 포장을 줄이고 있습니다.",
        "도시 사무실은 '에코 워킹타임'으로 전등을 1시간 일찍 끄는 에너지 절약을 실천하고 있습니다.",
        "산간 건설사는 친환경 자재를 활용한 구조물 설계를 추진하고 있습니다.",
        "바다 인근 영업소는 텀블러 사용자에게 리워드를 제공하는 친환경 마케팅을 운영하고 있습니다.",
        "도시 사무직으로 근무하며 사무실 내 커피 찌꺼기를 모아 화분 비료로 재활용하고 있습니다.",
        "산간 지역에서 재택근무하며 화면 밝기 자동 조절 기능을 통해 에너지 절약을 실천하고 있습니다.",
        "바다 인근에서 영업 활동 중 다회용 텀블러 사용을 생활화하며 쓰레기 배출을 줄이고 있습니다.",
        "농촌 현장직 근무자는 폐비닐 수거 활동을 정기적으로 실천하고 있습니다.",
        "도시 재택근무자는 매주 디지털 파일 정리와 클라우드 정리를 함께 수행하고 있습니다.",
        "영업직은 출장 시 대중교통을 이용하며 탄소배출 감소에 기여하고 있습니다.",
        "현장직으로 근무하며 점심시간에 자투리 나무를 활용해 소형 가구를 제작하고 있습니다.",
        "사무직으로 근무하며 매주 한 번 개인 텀블러 세척 캠페인을 진행하고 있습니다.",
        "바다 인근 재택근무자는 지역 해안 청소 행사에 참여하며 플로깅을 실천하고 있습니다.",
        "농촌 재택근무자는 계절마다 바뀌는 비건 식단 레시피를 온라인 사내 채널에 공유하고 있습니다.",
        "도시 사무직은 점심시간마다 텀블러를 사용하며 플라스틱 줄이기에 앞장서고 있습니다.",
        "산간 지역 재택근무자는 창문을 열고 자연채광을 활용해 전등 사용을 줄이고 있습니다.",
        "영업직은 출장 중 지하철을 이용하며 이동 중 친환경 콘텐츠를 소비하고 있습니다.",
        "바닷가 인근 재택근무자는 매일 아침 해변 플로깅을 실천하고 인증하고 있습니다.",
        "농촌의 현장직은 농기계 공회전을 줄이는 캠페인을 운영 중입니다.",
        "사무직은 종이 없는 회의를 위해 태블릿을 적극 활용하고 있습니다.",
        "산간 지역 사무직은 텀블러 사용률을 높이기 위해 매주 세척 캠페인을 운영하고 있습니다.",
        "도시 재택근무자는 업무 종료 후 콘센트를 뽑는 습관으로 에너지를 절약하고 있습니다.",
        "현장직은 자투리 목재를 재활용하여 작업 공간을 정비하고 있습니다.",
        "바다 인근 영업직은 친환경 용기 사용을 고객과 함께 실천하고 있습니다.",
        "농촌 재택근무자는 일주일에 한 번 지역 농산물만 활용한 식단을 구성하고 있습니다.",
        "사무직은 클라우드 폴더를 정리하여 디지털 탄소 줄이기 캠페인에 참여하고 있습니다.",
        "영업직은 다회용 컵을 챙겨 이동하며 일회용 쓰레기를 줄이고 있습니다.",
        "현장직은 매주 플라스틱 수거량을 기록하여 사내 보고에 활용하고 있습니다.",
        "산간 재택근무자는 중고 전자기기 교체 캠페인으로 전자폐기물을 줄이고 있습니다.",
        "도시 사무직은 '에코 출근 챌린지'를 통해 도보나 자전거로 출근하는 날을 정하고 있습니다.",
        "농촌 현장직은 태양광 전등으로 야간 작업 시 에너지 사용을 줄이고 있습니다.",
        "바다 인근 재택근무자는 친환경 세제를 사용해 세탁 후 오염수를 줄이고 있습니다.",
        "도시 재택근무자는 프린터 없는 업무 환경을 위해 PDF 문서 활용을 장려하고 있습니다.",
        "사무직은 종이컵 대신 개인 머그컵 사용을 정착시키기 위한 사내 캠페인을 운영하고 있습니다."
    ]
    
    # 2. 크롤링 데이터를 기반으로 한 챌린지 문장들
    crawled_challenges = []
    
    try:
        print("환경일보 뉴스를 크롤링하는 중...") # 메시지 수정
        news = fetch_environmental_news()
        for item in news:
            challenge = generate_challenge_from_news(item)
            if challenge:
                crawled_challenges.append({
                    "content": challenge["content"],
                    "metadata": challenge["metadata"]
                })
        
        print(f"크롤링 기반 챌린지 (환경일보 뉴스만) {len(crawled_challenges)}개 생성 완료") # 메시지 수정
        
    except Exception as e:
        print(f"크롤링 데이터 처리 중 오류 발생: {str(e)}")

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            if mode == "random":
                # 3. 고정 데이터와 크롤링 데이터를 구분하여 저장
                f.write("# 고정 데이터 챌린지\n")
                for _ in range(num_paragraphs // 2):  # 절반은 고정 데이터
                    num_sentences = random.randint(2, 5)
                    paragraph = " ".join(random.sample(random_sentences, num_sentences))
                    f.write(f"{clean_text(paragraph)}\n\n")
                
                f.write("\n# 크롤링 기반 챌린지\n")
                for challenge in crawled_challenges:
                    f.write(f"{challenge['content']}\n")
                    f.write(f"카테고리: {challenge['metadata']['category']}\n")
                    f.write(f"위치: {challenge['metadata'].get('location', 'N/A')}\n")
                    f.write(f"직종: {challenge['metadata'].get('job_type', 'N/A')}\n\n")
                
                print(f"랜덤 모드: {num_paragraphs}개 문단 저장 완료")
            else:
                print(f"오류: 지원하지 않는 모드입니다. ('fixed' 또는 'random' 사용)")

    except Exception as e:
        print(f"파일 생성 실패: {str(e)}")
    
    return {
        "fixed_challenges": random_sentences,
        "crawled_challenges": crawled_challenges
    }

if __name__ == "__main__":
    generate_challenge_docs()