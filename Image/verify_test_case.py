
test_cases = [
    # 개인 챌린지 
    { # 1
        "verificationId": 11,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/1_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 1,
        "challengeName": "텀블러 사용하기 챌린지",
        "expected": True
    },
    { # 2
        "verificationId": 12,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/1_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 1,
        "challengeName": "텀블러 사용하기 챌린지",
        "expected": False
    },
    { # 3
        "verificationId": 21,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/2_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.jpg",
        "challengeId": 2,
        "challengeName": "에코백 사용하기 챌린지",
        "expected": True
    },
    { # 4
        "verificationId": 22,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/2_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 2,
        "challengeName": "에코백 사용하기 챌린지",
        "expected": False
    },
    { # 5
        "verificationId": 31,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/3_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.jpg",
        "challengeId": 3,
        "challengeName": "장바구니 사용하기 챌린지",
        "expected": True
    },
    { # 6
        "verificationId": 32,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/3_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 3,
        "challengeName": "장바구니 사용하기 챌린지",
        "expected": False
    },
    { # 7
        "verificationId": 41,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/4_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.jpg",
        "challengeId": 4,
        "challengeName": "자전거 타기 챌린지",
        "expected": True
    },
    { # 8
        "verificationId": 42,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/4_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 4,
        "challengeName": "자전거 타기 챌린지",
        "expected": False
    },
    { # 9
        "verificationId": 51,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/5_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.jpg",
        "challengeId": 5,
        "challengeName": "대중교통 이용 챌린지",
        "expected": True
    },
    { # 10
        "verificationId": 52,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/5_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 5,
        "challengeName": "대중교통 이용 챌린지",
        "expected": False
    },
    { # 11
        "verificationId": 61,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/6_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.jpg",
        "challengeId": 6,
        "challengeName": "샐러드/채식 식단 먹기 챌린지",
        "expected": True
    },
    { # 12
        "verificationId": 62,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/6_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 6,
        "challengeName": "샐러드/채식 식단 먹기 챌린지",
        "expected": False
    },
    { # 13
        "verificationId": 71,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/7_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 7,
        "challengeName": "음식 남기지 않기 챌린지",
        "expected": True
    },
    { # 14
        "verificationId": 72,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/7_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 7,
        "challengeName": "음식 남기지 않기 챌린지",
        "expected": False
    },
    { # 15
        "verificationId": 81,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/8_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.jpg",
        "challengeId": 8,
        "challengeName": "계단 이용하기 챌린지",
        "expected": True
    },
    { # 16
        "verificationId": 82,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/8_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 8,
        "challengeName": "계단 이용하기 챌린지",
        "expected": False
    },
    { # 17
        "verificationId": 91,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/9_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 9,
        "challengeName": "재활용 분리수거 챌린지",
        "expected": True
    },
    { # 18
        "verificationId": 92,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/9_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 9,
        "challengeName": "재활용 분리수거 챌린지",
        "expected": False
    },
    { # 19
        "verificationId": 101,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/10_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 10,
        "challengeName": "손수건 사용 챌린지",
        "expected": True
    },
    { # 20
        "verificationId": 102,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/10_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 10,
        "challengeName": "손수건 사용 챌린지",
        "expected": False
    },
    { # 21
        "verificationId": 111,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/11_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.jpg",
        "challengeId": 11,
        "challengeName": "쓰레기 줍기 챌린지",
        "expected": True
    },
    { # 22
        "verificationId": 112,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/11_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 11,
        "challengeName": "쓰레기 줍기 챌린지",
        "expected": False
    },
    { # 23
        "verificationId": 121,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/12_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 12,
        "challengeName": "안쓰는 전기 플러그 뽑기 챌린지",
        "expected": True
    },
    { # 24
        "verificationId": 122,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/12_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 12,
        "challengeName": "안쓰는 전기 플러그 뽑기 챌린지",
        "expected": False
    },
    { # 25
        "verificationId": 131,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/13_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.jpg",
        "challengeId": 13,
        "challengeName": "고체 비누 사용 챌린지",
        "expected": True
    },
    { # 26
        "verificationId": 132,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/13_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 13,
        "challengeName": "고체 비누 사용 챌린지",
        "expected": False
    },
    { # 27
        "verificationId": 141,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/14_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 14,
        "challengeName": "하루 만보 걷기 챌린지",
        "expected": True
    },
    { # 28
        "verificationId": 142,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/14_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.jpg",
        "challengeId": 14,
        "challengeName": "하루 만보 걷기 챌린지",
        "expected": False
    },
    { # 29
        "verificationId": 151,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/15_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.jpg",
        "challengeId": 15,
        "challengeName": "도시락 싸먹기 챌린지",
        "expected": True
    },
    { # 30
        "verificationId": 152,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/15_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 15,
        "challengeName": "도시락 싸먹기 챌린지",
        "expected": False
    },
    { # 31
        "verificationId": 161,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/16_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.jpg",
        "challengeId": 16,
        "challengeName": "작은 텃밭 가꾸기 챌린지",
        "expected": True
    },
    { # 32
        "verificationId": 162,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/16_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 16,
        "challengeName": "작은 텃밭 가꾸기 챌린지",
        "expected": False
    },
    { # 33
        "verificationId": 171,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/17_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 17,
        "challengeName": "반려 식물 인증 챌린지",
        "expected": True
    },
    { # 34
        "verificationId": 172,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/17_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 17,
        "challengeName": "반려 식물 인증 챌린지",
        "expected": False
    },
    { # 35
        "verificationId": 181,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/18_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.jpg",
        "challengeId": 18,
        "challengeName": "전자 영수증 받기 챌린지",
        "expected": True
    },
    { # 36
        "verificationId": 182,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/18_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 18,
        "challengeName": "전자 영수증 받기 챌린지",
        "expected": False
    },
    { # 37
        "verificationId": 191,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/19_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 19,
        "challengeName": "친환경 인증 마크 상품 구매하기 챌린지",
        "expected": True
    },
    { # 38
        "verificationId": 192,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/19_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 19,
        "challengeName": "친환경 인증 마크 상품 구매하기 챌린지",
        "expected": False
    },
    { # 39
        "verificationId": 201,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/20_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.jpg",
        "challengeId": 20,
        "challengeName": "다회용기 사용하기 챌린지",
        "expected": True
    },
    { # 40
        "verificationId": 202,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/20_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 20,
        "challengeName": "다회용기 사용하기 챌린지",
        "expected": False
    },
    { # 41
        "verificationId": 211,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/21_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 21,
        "challengeName": "대나무 칫솔 사용하기 챌린지",
        "expected": True
    },
    { # 42
        "verificationId": 212,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/21_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 21,
        "challengeName": "대나무 칫솔 사용하기 챌린지",
        "expected": False
    },
    # 이벤트 챌린지 -> 멀티 프롬프팅 필요 + challengeId를 기준으로 나누면 될 듯 
    { # 43
        "verificationId": 110,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/1_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 1,
        "challengeName": "SNS에 습지 보호 캠페인 알리기",
        "expected": True
    },
    { # 44
        "verificationId": 120,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/1_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 1,
        "challengeName": "SNS에 습지 보호 캠페인 알리기",
        "expected": False
    },
    { # 45 
        "verificationId": 310,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/3_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 3,
        "challengeName": "생명의 물을 지켜요! 생활 속 절수+물길 정화 캠페인",
        "expected": True
    },
    { # 46
        "verificationId": 320,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/3_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 3,
        "challengeName": "생명의 물을 지켜요! 생활 속 절수+물길 정화 캠페인",
        "expected": False
    },
    { # 47
        "verificationId": 410,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/4_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 4,
        "challengeName": "오늘 내가 심은 나무 한 그루",
        "expected": True
    },
    { # 48
        "verificationId": 420,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/4_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 4,
        "challengeName": "오늘 내가 심은 나무 한 그루",
        "expected": False
    },
    { # 49
        "verificationId": 510,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/5_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 5,
        "challengeName": "지구야, 미안하고 고마워 🌍 편지 쓰기 챌린지",
        "expected": True
    },
    { # 50
        "verificationId": 520,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/5_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 5,
        "challengeName": "지구야, 미안하고 고마워 🌍 편지 쓰기 챌린지",
        "expected": False
    },
    { # 51
        "verificationId": 710,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/7_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 7,
        "challengeName": "착한 소비, 지구도 사람도 웃게 해요",
        "expected": True
    },
    { # 52
        "verificationId": 720,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/7_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 7,
        "challengeName": "착한 소비, 지구도 사람도 웃게 해요",
        "expected": False
    },
    { # 53
        "verificationId": 810,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/8_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 8,
        "challengeName": "오늘은 바다를 위해 한 걸음",
        "expected": True
    },
    { # 54
        "verificationId": 820,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/8_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 8,
        "challengeName": "오늘은 바다를 위해 한 걸음",
        "expected": False
    },
    { # 55
        "verificationId": 910,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/9_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 9,
        "challengeName": "나의 환경 한 가지 실천 DAY",
        "expected": True
    },
    { # 56
        "verificationId": 920,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/9_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 9,
        "challengeName": "나의 환경 한 가지 실천 DAY",
        "expected": False
    },
    { # 57
        "verificationId": 1010,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/10_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 10,
        "challengeName": "양치컵 하나로 지구를 살려요!",
        "expected": True
    },
    { # 58
        "verificationId": 1020,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/10_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 10,
        "challengeName": "양치컵 하나로 지구를 살려요!",
        "expected": False
    },
    { # 59
        "verificationId": 1110,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/11_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 11,
        "challengeName": "호랑이를 지켜요! 숲을 위한 하루",
        "expected": True
    },
    { # 60
        "verificationId": 1120,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/11_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 11,
        "challengeName": "호랑이를 지켜요! 숲을 위한 하루",
        "expected": False
    },
    { # 61
        "verificationId": 1210,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/12_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 12,
        "challengeName": "꺼주세요 1시간! 에너지를 아끼는 시간 OFF",
        "expected": True
    },
    { # 62
        "verificationId": 1220,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/12_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 12,
        "challengeName": "꺼주세요 1시간! 에너지를 아끼는 시간 OFF",
        "expected": False
    },
    { # 63
        "verificationId": 1310,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/13_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 13,
        "challengeName": "버리지 마세요! 오늘은 자원순환 챌린지",
        "expected": True
    },
    { # 64
        "verificationId": 1320,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/13_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 13,
        "challengeName": "버리지 마세요! 오늘은 자원순환 챌린지",
        "expected": False
    },
    { # 65
        "verificationId": 1410,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/14_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 14,
        "challengeName": "오늘은 걷거나 타세요! Car-Free 실천 챌린지",
        "expected": True
    },
    { # 66
        "verificationId": 1420,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/14_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 14,
        "challengeName": "오늘은 걷거나 타세요! Car-Free 실천 챌린지",
        "expected": False
    },
    { # 67
        "verificationId": 1510,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/15_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 15,
        "challengeName": "기후재난 이야기 공유 챌린지",
        "expected": True
    },
    { # 68
        "verificationId": 1520,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/15_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 15,
        "challengeName": "기후재난 이야기 공유 챌린지",
        "expected": False
    },
    { # 69
        "verificationId": 1610,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/16_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 16,
        "challengeName": "오늘은 비건 한 끼, 지구와 나를 위한 식사",
        "expected": True
    },
    { # 70
        "verificationId": 1620,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/16_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 16,
        "challengeName": "오늘은 비건 한 끼, 지구와 나를 위한 식사",
        "expected": False
    },
    { # 71
        "verificationId": 210,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/2_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 2,
        "challengeName": "해양 정화로 고래를 지켜요",
        "expected": True
    },
    { # 72
        "verificationId": 220,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/2_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 2,
        "challengeName": "해양 정화로 고래를 지켜요",
        "expected": False
    },
    { # 73
        "verificationId": 610,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/6_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 6,
        "challengeName": "음식물도 순환돼요! 퇴비 챌린지",
        "expected": True
    },
    { # 74
        "verificationId": 620,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/6_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 6,
        "challengeName": "음식물도 순환돼요! 퇴비 챌린지",
        "expected": False
    },
    { # 75
        "verificationId": 1710,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/17_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 17,
        "challengeName": "한 뼘의 텃밭, 농민의 마음을 심어요",
        "expected": True
    },
    { # 76
        "verificationId": 1720,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/17_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 17,
        "challengeName": "한 뼘의 텃밭, 농민의 마음을 심어요",
        "expected": False
    }
]

'''
{ # 
        "verificationId": 10,
        "type": "PERSONAL",
        "imageUrl": "",
        "challengeId": ,
        "challengeName": " 챌린지",
        "expected": True
    },
    { # 
        "verificationId": 20,
        "type": "PERSONAL",
        "imageUrl": "",
        "challengeId": ,
        "challengeName": " 챌린지",
        "expected": False
    },
'''
