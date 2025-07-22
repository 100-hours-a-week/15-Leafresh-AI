import json
import random
import re
import copy
import datetime
from pathlib import Path

# 데이터 생성 관련 상수 및 샘플
locations = ["도시", "바닷가", "산", "농촌"]
work_types = ["사무직", "현장직", "영업직", "재택근무"]
categories = [
    "제로웨이스트", "플로깅", "탄소발자국", "에너지 절약",
    "업사이클", "문화 공유", "디지털 탄소", "비건"
]
free_text_inputs = [
    "요즘 너무 피곤해서 아무것도 하기 싫어.",
    "환경을 위해 내가 할 수 있는 게 뭐가 있을까?",
    "친구랑 같이 할만한 챌린지 추천해줘.",
    "집에서 할 수 있는 친환경 활동 알려줘.",
    "오늘 날씨가 너무 더워서 힘들어.",
    "회사에서 쉽게 실천할 수 있는 환경 챌린지 뭐 있어?",
    "반려동물이랑 같이 할 수 있는 활동 알려줘.",
    "스트레스 받을 때 할만한 친환경 챌린지 추천해줘."
]
free_text_recommends = [
    "상황에 맞는 친환경 챌린지를 추천해 드릴게요!",
    "이럴 때 실천하면 좋은 친환경 챌린지 3가지를 알려드릴게요!",
    "기분 전환과 환경 보호를 동시에 할 수 있는 챌린지입니다!",
    "일상에서 쉽게 실천할 수 있는 친환경 챌린지를 추천합니다!",
    "함께하면 더 즐거운 친환경 챌린지 3가지를 소개할게요!"
]
free_text_challenges = [
    {"title": "1. 햇볕 쬐며 10분 산책하기", "description": "점심시간에 잠시 공원을 걸으며 비타민 D를 보충하고 기분을 전환해보세요."},
    {"title": "2. 건강한 채소 주스 마시기", "description": "신선한 채소와 과일로 만든 주스로 몸에 생기를 불어넣어요."},
    {"title": "3. 숙면을 위한 디지털 디톡스", "description": "잠들기 1시간 전에는 스마트폰을 멀리하고 편안한 휴식을 취해보세요."},
    {"title": "4. 환경 다큐멘터리 시청하기", "description": "환경을 주제로 한 다큐멘터리를 감상해보세요."},
    {"title": "5. 플로깅 참여하기", "description": "산책이나 운동할 때 주변 쓰레기를 주워보세요."},
    {"title": "6. 텀블러 사용하기", "description": "일회용 컵 대신 텀블러를 사용해보세요."},
    {"title": "7. 채식 식단 도전하기", "description": "일주일에 하루는 채식 식단에 도전해보세요."},
    {"title": "8. 중고 거래 앱 이용하기", "description": "필요한 물건은 중고 거래 앱에서 찾아보세요."},
    {"title": "9. 다회용기 사용하기", "description": "포장 음식 주문 시 다회용기를 사용해보세요."},
    {"title": "10. 실내 적정 온도 유지하기", "description": "여름엔 26도, 겨울엔 20도로 실내 온도를 유지해보세요."},
    {"title": "11. 반려동물과 플로깅하기", "description": "반려동물과 산책하며 쓰레기를 주워보세요."},
    {"title": "12. 업사이클링 공예 도전하기", "description": "집에 있는 재료로 업사이클링 소품을 만들어보세요."}
]
feedback_titles = [
    "텀블러 사용하기", "음식 남기지 않기", "대중교통 이용하기", "계단 이용하기", "플로깅 참여하기", "비건 식단 도전하기",
    "쓰레기 줍기", "에코백 사용하기", "자전거 타기", "손수건 사용", "재활용 분리수거", "도시락 싸먹기"
]
group_titles = [
    "비건데이", "플로깅", "도시락 싸먹기", "에코백 사용하기", "종이컵 대신 머그컵", "환경 다큐 시청"
]
success_emojis = ["🌱", "🎉", "👍", "✨"]
fail_emojis = ["😅", "😢", "😭"]
encourage_emojis = ["💪", "💖", "😊", "👏"]

challenge_title_templates = [
    "{num}. {category} 실천하기",
    "{num}. {category} 챌린지 도전",
    "{num}. {category} 관련 활동",
    "{num}. {category} 습관 만들기",
    "{num}. {category} 미션",
    "{num}. {category} 함께 해요"
]
challenge_desc_templates = [
    "{location}에서 {work_type}이 할 수 있는 {category} 챌린지입니다.",
    "{work_type} 환경에서 실천할 수 있는 {category} 활동을 해보세요.",
    "동료들과 {category} 팁을 공유하며 함께 실천해보세요.",
    "오늘은 {category}에 한 번 도전해보는 건 어때요?",
    "작은 실천이 큰 변화를 만듭니다! {category}에 도전해보세요.",
    "{category}로 환경을 지키는 멋진 하루를 만들어보세요."
]
challenge_recommend_templates = [
    "{location}, {work_type}, {category}에 어울리는 친환경 챌린지 3가지를 추천합니다!",
    "{location}에서 {work_type}에게 딱 맞는 {category} 챌린지 세 가지!",
    "{work_type}이(가) {location}에서 실천할 수 있는 {category} 미션을 소개할게요.",
    "{category}에 관심 있는 {work_type}에게 추천하는 챌린지 TOP 3!",
    "{location} 라이프스타일에 맞는 {category} 챌린지, 지금 시작해보세요!"
]
free_text_recommend_templates = [
    "상황에 맞는 친환경 챌린지를 추천해 드릴게요!",
    "이럴 때 실천하면 좋은 친환경 챌린지 3가지를 알려드릴게요!",
    "기분 전환과 환경 보호를 동시에 할 수 있는 챌린지입니다!",
    "일상에서 쉽게 실천할 수 있는 친환경 챌린지를 추천합니다!",
    "함께하면 더 즐거운 친환경 챌린지 3가지를 소개할게요!",
    "오늘은 이런 챌린지에 도전해보는 건 어떨까요?",
    "작은 실천이 큰 변화를 만듭니다!",
    "환경을 생각하는 당신께 드리는 챌린지 제안입니다!"
]

def pick_emoji(pool):
    return random.choice(pool)

def random_date(start_year=2025, end_year=2026):
    start = datetime.date(start_year, 1, 1)
    end = datetime.date(end_year, 12, 31)
    delta = end - start
    random_days = random.randint(0, delta.days)
    d = start + datetime.timedelta(days=random_days)
    return d.strftime("%Y-%m-%dT%H:%M:%S")

def generate_challenge_items(existing, target):
    combos = [(l, w, c) for l in locations for w in work_types for c in categories]
    used = set((item["input"].strip(),) for item in existing)
    new_items = []
    for l, w, c in combos:
        key = f"{l}, {w}, {c}"
        if (key,) in used:
            continue
        # 다양한 템플릿 랜덤 적용
        recommend = random.choice(challenge_recommend_templates).format(location=l, work_type=w, category=c)
        challenges = [
            {
                "title": random.choice(challenge_title_templates).format(num=idx+1, category=c),
                "description": random.choice(challenge_desc_templates).format(location=l, work_type=w, category=c)
            }
            for idx in range(3)
        ]
        # 무조건 줄바꿈 문자열 포맷
        challenge_lines = [f"{ch['title']}: {ch['description']}" for ch in challenges]
        output_text = recommend + "\n" + "\n".join(challenge_lines)
        new_items.append({
            "instruction": "너는 챌린지 추천 챗봇이야. 사용자가 선택한 '위치, 직업, 카테고리'에 맞춰 구체적인 친환경 챌린지 3가지를 줄바꿈(\\n)으로 구분해서 추천해줘.",
            "input": key,
            "output": output_text
        })
        if len(existing) + len(new_items) >= target:
            break
    return existing + new_items[:max(0, target - len(existing))]

def generate_feedback_items(existing, target):
    new_items = []
    for i in range(target - len(existing)):
        member_id = random.randint(100, 999)
        n_personal = random.randint(0, 3)
        personal = []
        for _ in range(n_personal):
            title = random.choice(feedback_titles)
            is_success = random.choice([True, False])
            personal.append({
                "id": random.randint(1, 999),
                "title": title,
                "isSuccess": is_success
            })
        n_group = random.randint(0, 2)
        group = []
        for _ in range(n_group):
            title = random.choice(group_titles)
            start = random_date()
            end = random_date()
            if end < start:
                start, end = end, start
            n_sub = random.randint(0, 3)
            submissions = []
            for _ in range(n_sub):
                submissions.append({
                    "isSuccess": random.choice([True, False]),
                    "submittedAt": random_date()
                })
            group.append({
                "id": random.randint(100, 999),
                "title": title,
                "startDate": start,
                "endDate": end,
                "submissions": submissions
            })
        inp = json.dumps({
            "memberId": member_id,
            "personalChallenges": personal,
            "groupChallenges": group
        }, ensure_ascii=False)
        successes = [c["title"] for c in personal if c["isSuccess"]]
        fails = [c["title"] for c in personal if not c["isSuccess"]]
        group_titles_done = [g["title"] for g in group if any(s["isSuccess"] for s in g["submissions"])]
        group_titles_none = [g["title"] for g in group if not g["submissions"]]
        msg = []
        if successes:
            msg.append(f"{', '.join(successes)} 성공! 정말 멋져요. {pick_emoji(success_emojis)}")
        if fails:
            msg.append(f"{', '.join(fails)}는(은) 조금 아쉬웠지만, 도전 자체가 의미 있어요. {pick_emoji(fail_emojis)}")
        if group_titles_done:
            msg.append(f"{', '.join(group_titles_done)} 챌린지에도 꾸준히 참여하셨네요! 대단해요. {pick_emoji(success_emojis)}")
        if group_titles_none:
            msg.append(f"{', '.join(group_titles_none)} 챌린지는 아직 참여 기록이 없어요. 다음엔 꼭 도전해봐요! {pick_emoji(encourage_emojis)}")
        if not msg:
            msg.append(f"이번 주는 챌린지 활동이 없었네요. 다음엔 새로운 도전을 기대할게요! {pick_emoji(encourage_emojis)}")
        msg_joined = " ".join(msg)
        # 문장 단위로 300자 이내로 최대한 길게 포함, 최소 70자 이상
        sentences = re.split(r'(?<=[.!?]) +', msg_joined)
        out = ""
        for s in sentences:
            if len(out) + len(s) > 300:
                break
            out += s + " "
        out = out.strip()
        # 최소 70자 이상이 되도록 추가
        if len(out) < 70:
            for s in sentences[len(out.split('. ')):]:
                if len(out) + len(s) > 300:
                    break
                out += " " + s
                if len(out) >= 70:
                    break
            out = out.strip()
        # 그래도 70자 미만이면 그냥 70자까지 자르기
        if len(out) < 70:
            out = (msg_joined + " " + out)[:70]
        new_items.append({
            "instruction": "너는 피드백 어시스턴트야. 아래와 같은 JSON 입력을 받으면, 사용자의 챌린지 활동을 요약해서 칭찬과 격려를 해줘. 한글로, 300자 이내로 답해.",
            "input": inp,
            "output": out
        })
    return existing + new_items

def generate_free_text_items(existing, target):
    new_items = []
    for i in range(target - len(existing)):
        inp = random.choice(free_text_inputs)
        recommend = random.choice(free_text_recommend_templates)
        challenges = random.sample(free_text_challenges, 3)
        challenge_lines = [
            f"{idx+1}. {c['title'].split('. ', 1)[-1]}: {c['description']}" for idx, c in enumerate(challenges)
        ]
        output_text = recommend + "\n" + "\n".join(challenge_lines)
        new_items.append({
            "instruction": "너는 사용자와 자유롭게 대화하며 대화의 맥락에 맞는 친환경 챌린지 3가지를 줄바꿈(\\n)으로 구분해서 추천하는 챗봇이야.",
            "input": inp,
            "output": output_text
        })
    return existing + new_items

def parse_output_to_object(data):
    for item in data:
        if "output" in item:
            try:
                if isinstance(item["output"], str) and item["output"].strip().startswith("{") and item["output"].strip().endswith("}"):
                    item["output"] = json.loads(item["output"])
            except Exception as e:
                print("Error parsing output:", item["output"])
                raise e
    return data

def fix_challenge_title_numbering(data):
    for item in data:
        out = item.get("output")
        if isinstance(out, dict) and "challenges" in out:
            for idx, challenge in enumerate(out["challenges"], 1):
                old_title = challenge.get("title", "")
                new_title = re.sub(r"^\d+\.\s*", "", old_title)
                challenge["title"] = f"{idx}. {new_title}"
    return data

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def force_output_to_string(data):
    for item in data:
        # 피드백 어시스턴트 유형은 건드리지 않음
        if '피드백 어시스턴트' in item.get('instruction', ''):
            continue
        out = item["output"]
        # 이미 dict면 그대로 직렬화
        if isinstance(out, dict):
            item["output"] = json.dumps(out, ensure_ascii=False)
        # 문자열인데 JSON 오브젝트가 아니면 text로 감싸기
        elif isinstance(out, str):
            s = out.strip()
            if s.startswith("{") and s.endswith("}"):
                # 이미 JSON 문자열이면 그대로 둠
                item["output"] = out
            else:
                # 일반 텍스트면 {'text': ...}로 감싸서 JSON 문자열로 변환
                item["output"] = json.dumps({"text": out}, ensure_ascii=False)
    return data

def add_newlines_to_output(data):
    for item in data:
        try:
            out = item["output"]
            if isinstance(out, str) and out.strip().startswith("{") and out.strip().endswith("}"):
                obj = json.loads(out)
                # recommend 줄바꿈
                if "recommend" in obj and not obj["recommend"].endswith("\n"):
                    obj["recommend"] = obj["recommend"].rstrip() + "\n"
                # challenges 각 항목 줄바꿈
                if "challenges" in obj:
                    for ch in obj["challenges"]:
                        if "title" in ch and not ch["title"].endswith("\n"):
                            ch["title"] = ch["title"].rstrip() + "\n"
                        if "description" in ch and not ch["description"].endswith("\n"):
                            ch["description"] = ch["description"].rstrip() + "\n"
                item["output"] = json.dumps(obj, ensure_ascii=False)
        except Exception:
            continue
    return data

def main():
    # 기존 데이터 로드
    DATASET_PATH = Path("multitask_dataset_2000+300.json")
    with open(DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    # 유형별 목표 개수
    TARGET_SIZE = 1800
    n_types = 3
    per_type = TARGET_SIZE // n_types
    extra = TARGET_SIZE - per_type * n_types
    type_targets = [per_type] * n_types
    for i in range(extra):
        type_targets[i] += 1

    # 기존 데이터 분류
    challenge_items = [x for x in data if '챌린지 추천' in x['instruction']]
    free_text_items = [x for x in data if '자유롭게 대화' in x['instruction']]
    feedback_items = [x for x in data if '피드백 어시스턴트' in x['instruction']]

    # 부족분 생성 (항상 목표 개수만큼 맞추기)
    challenge_items_final = (challenge_items + generate_challenge_items([], type_targets[0]))[:type_targets[0]]
    free_text_items_final = (free_text_items + generate_free_text_items([], type_targets[1]))[:type_targets[1]]
    feedback_items_final = (feedback_items + generate_feedback_items([], type_targets[2]))[:type_targets[2]]

    all_items = challenge_items_final + free_text_items_final + feedback_items_final
    random.shuffle(all_items)

    # output 오브젝트 변환
    data_obj = parse_output_to_object(copy.deepcopy(all_items))
    data_numbered = fix_challenge_title_numbering(copy.deepcopy(data_obj))
    data_stringified = force_output_to_string(data_numbered)
    data_stringified = add_newlines_to_output(data_stringified)  # 줄바꿈 추가
    save_json(data_stringified, "multitask_dataset_v3.json")
    print(f"Done. Saved to Text/LLM/multitask_dataset_v3.json")

if __name__ == "__main__":
    main() 