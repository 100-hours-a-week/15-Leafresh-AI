
test_cases = [
    # 개인 챌린지 
    { # 1
        "verificationId": 11,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/1_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 1,
        "challengeName": "텀블러 사용하기 챌린지",
        "challengeInfo": "",
        "expected": True
    },
    { # 2
        "verificationId": 12,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/1_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 1,
        "challengeName": "텀블러 사용하기 챌린지",
        "challengeInfo": "",
        "expected": False
    },
    { # 3
        "verificationId": 21,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/2_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.jpg",
        "challengeId": 2,
        "challengeName": "에코백 사용하기 챌린지",
        "challengeInfo": "",
        "expected": True
    },
    { # 4
        "verificationId": 22,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/2_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 2,
        "challengeName": "에코백 사용하기 챌린지",
        "challengeInfo": "",
        "expected": False
    },
    { # 5
        "verificationId": 31,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/3_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.jpg",
        "challengeId": 3,
        "challengeName": "장바구니 사용하기 챌린지",
        "challengeInfo": "",
        "expected": True
    },
    { # 6
        "verificationId": 32,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/3_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 3,
        "challengeName": "장바구니 사용하기 챌린지",
        "challengeInfo": "",
        "expected": False
    },
    { # 7
        "verificationId": 41,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/4_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.jpg",
        "challengeId": 4,
        "challengeName": "자전거 타기 챌린지",
        "challengeInfo": "",
        "expected": True
    },
    { # 8
        "verificationId": 42,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/4_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 4,
        "challengeName": "자전거 타기 챌린지",
        "challengeInfo": "",
        "expected": False
    },
    { # 9
        "verificationId": 51,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/5_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.jpg",
        "challengeId": 5,
        "challengeName": "대중교통 이용 챌린지",
        "challengeInfo": "",
        "expected": True
    },
    { # 10
        "verificationId": 52,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/5_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 5,
        "challengeName": "대중교통 이용 챌린지",
        "challengeInfo": "",
        "expected": False
    },
    { # 11
        "verificationId": 61,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/6_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.jpg",
        "challengeId": 6,
        "challengeName": "샐러드/채식 식단 먹기 챌린지",
        "challengeInfo": "",
        "expected": True
    },
    { # 12
        "verificationId": 62,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/6_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 6,
        "challengeName": "샐러드/채식 식단 먹기 챌린지",
        "challengeInfo": "",
        "expected": False
    },
    { # 13
        "verificationId": 71,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/7_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 7,
        "challengeName": "음식 남기지 않기 챌린지",
        "challengeInfo": "",
        "expected": True
    },
    { # 14
        "verificationId": 72,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/7_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 7,
        "challengeName": "음식 남기지 않기 챌린지",
        "challengeInfo": "",
        "expected": False
    },
    { # 15
        "verificationId": 81,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/8_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.jpg",
        "challengeId": 8,
        "challengeName": "계단 이용하기 챌린지",
        "challengeInfo": "",
        "expected": True
    },
    { # 16
        "verificationId": 82,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/8_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 8,
        "challengeName": "계단 이용하기 챌린지",
        "challengeInfo": "",
        "expected": False
    },
    { # 17
        "verificationId": 91,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/9_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 9,
        "challengeName": "재활용 분리수거 챌린지",
        "challengeInfo": "",
        "expected": True
    },
    { # 18
        "verificationId": 92,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/9_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 9,
        "challengeName": "재활용 분리수거 챌린지",
        "challengeInfo": "",
        "expected": False
    },
    { # 19
        "verificationId": 101,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/10_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 10,
        "challengeName": "손수건 사용 챌린지",
        "challengeInfo": "",
        "expected": True
    },
    { # 20
        "verificationId": 102,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/10_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 10,
        "challengeName": "손수건 사용 챌린지",
        "challengeInfo": "",
        "expected": False
    },
    { # 21
        "verificationId": 111,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/11_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.jpg",
        "challengeId": 11,
        "challengeName": "쓰레기 줍기 챌린지",
        "challengeInfo": "",
        "expected": True
    },
    { # 22
        "verificationId": 112,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/11_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 11,
        "challengeName": "쓰레기 줍기 챌린지",
        "challengeInfo": "",
        "expected": False
    },
    { # 23
        "verificationId": 121,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/12_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 12,
        "challengeName": "안쓰는 전기 플러그 뽑기 챌린지",
        "challengeInfo": "",
        "expected": True
    },
    { # 24
        "verificationId": 122,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/12_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 12,
        "challengeName": "안쓰는 전기 플러그 뽑기 챌린지",
        "challengeInfo": "",
        "expected": False
    },
    { # 25
        "verificationId": 131,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/13_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.jpg",
        "challengeId": 13,
        "challengeName": "고체 비누 사용 챌린지",
        "challengeInfo": "",
        "expected": True
    },
    { # 26
        "verificationId": 132,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/13_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 13,
        "challengeName": "고체 비누 사용 챌린지",
        "challengeInfo": "",
        "expected": False
    },
    { # 27
        "verificationId": 141,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/14_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 14,
        "challengeName": "하루 만보 걷기 챌린지",
        "challengeInfo": "",
        "expected": True
    },
    { # 28
        "verificationId": 142,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/14_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.jpg",
        "challengeId": 14,
        "challengeName": "하루 만보 걷기 챌린지",
        "challengeInfo": "",
        "expected": False
    },
    { # 29
        "verificationId": 151,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/15_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.jpg",
        "challengeId": 15,
        "challengeName": "도시락 싸먹기 챌린지",
        "challengeInfo": "",
        "expected": True
    },
    { # 30
        "verificationId": 152,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/15_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 15,
        "challengeName": "도시락 싸먹기 챌린지",
        "challengeInfo": "",
        "expected": False
    },
    { # 31
        "verificationId": 161,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/16_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.jpg",
        "challengeId": 16,
        "challengeName": "작은 텃밭 가꾸기 챌린지",
        "challengeInfo": "",
        "expected": True
    },
    { # 32
        "verificationId": 162,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/16_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 16,
        "challengeName": "작은 텃밭 가꾸기 챌린지",
        "challengeInfo": "",
        "expected": False
    },
    { # 33
        "verificationId": 171,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/17_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 17,
        "challengeName": "반려 식물 인증 챌린지",
        "challengeInfo": "",
        "expected": True
    },
    { # 34
        "verificationId": 172,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/17_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 17,
        "challengeName": "반려 식물 인증 챌린지",
        "challengeInfo": "",
        "expected": False
    },
    { # 35
        "verificationId": 181,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/18_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.jpg",
        "challengeId": 18,
        "challengeName": "전자 영수증 받기 챌린지",
        "challengeInfo": "",
        "expected": True
    },
    { # 36
        "verificationId": 182,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/18_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 18,
        "challengeName": "전자 영수증 받기 챌린지",
        "challengeInfo": "",
        "expected": False
    },
    { # 37
        "verificationId": 191,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/19_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 19,
        "challengeName": "친환경 인증 마크 상품 구매하기 챌린지",
        "challengeInfo": "",
        "expected": True
    },
    { # 38
        "verificationId": 192,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/19_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 19,
        "challengeName": "친환경 인증 마크 상품 구매하기 챌린지",
        "challengeInfo": "",
        "expected": False
    },
    { # 39
        "verificationId": 201,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/20_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.jpg",
        "challengeId": 20,
        "challengeName": "다회용기 사용하기 챌린지",
        "challengeInfo": "",
        "expected": True
    },
    { # 40
        "verificationId": 202,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/20_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 20,
        "challengeName": "다회용기 사용하기 챌린지",
        "challengeInfo": "",
        "expected": False
    },
    { # 41
        "verificationId": 211,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/21_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 21,
        "challengeName": "대나무 칫솔 사용하기 챌린지",
        "challengeInfo": "",
        "expected": True
    },
    { # 42
        "verificationId": 212,
        "type": "PERSONAL",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/21_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 21,
        "challengeName": "대나무 칫솔 사용하기 챌린지",
        "challengeInfo": "",
        "expected": False
    },
    # 이벤트 챌린지 -> 멀티 프롬프팅 필요 + challengeId를 기준으로 나누면 될 듯 
    { # 43
        "verificationId": 110,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/1_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 1,
        "challengeName": "SNS에 습지 보호 캠페인 알리기",
        "challengeInfo": "",
        "expected": True
    },
    { # 44
        "verificationId": 120,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/1_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 1,
        "challengeName": "SNS에 습지 보호 캠페인 알리기",
        "challengeInfo": "",
        "expected": False
    },
    { # 45 
        "verificationId": 310,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/3_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 3,
        "challengeName": "생명의 물을 지켜요! 생활 속 절수+물길 정화 캠페인",
        "challengeInfo": "",
        "expected": True
    },
    { # 46
        "verificationId": 320,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/3_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 3,
        "challengeName": "생명의 물을 지켜요! 생활 속 절수+물길 정화 캠페인",
        "challengeInfo": "",
        "expected": False
    },
    { # 47
        "verificationId": 410,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/4_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 4,
        "challengeName": "오늘 내가 심은 나무 한 그루",
        "challengeInfo": "",
        "expected": True
    },
    { # 48
        "verificationId": 420,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/4_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 4,
        "challengeName": "오늘 내가 심은 나무 한 그루",
        "challengeInfo": "",
        "expected": False
    },
    { # 49
        "verificationId": 510,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/5_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 5,
        "challengeName": "지구야, 미안하고 고마워 🌍 편지 쓰기 챌린지",
        "challengeInfo": "",
        "expected": True
    },
    { # 50
        "verificationId": 520,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/5_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 5,
        "challengeName": "지구야, 미안하고 고마워 🌍 편지 쓰기 챌린지",
        "challengeInfo": "",
        "expected": False
    },
    { # 51
        "verificationId": 710,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/7_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 7,
        "challengeName": "착한 소비, 지구도 사람도 웃게 해요",
        "challengeInfo": "",
        "expected": True
    },
    { # 52
        "verificationId": 720,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/7_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 7,
        "challengeName": "착한 소비, 지구도 사람도 웃게 해요",
        "challengeInfo": "",
        "expected": False
    },
    { # 53
        "verificationId": 810,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/8_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 8,
        "challengeName": "오늘은 바다를 위해 한 걸음",
        "challengeInfo": "",
        "expected": True
    },
    { # 54
        "verificationId": 820,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/8_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 8,
        "challengeName": "오늘은 바다를 위해 한 걸음",
        "challengeInfo": "",
        "expected": False
    },
    { # 55
        "verificationId": 910,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/9_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 9,
        "challengeName": "나의 환경 한 가지 실천 DAY",
        "challengeInfo": "",
        "expected": True
    },
    { # 56
        "verificationId": 920,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/9_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 9,
        "challengeName": "나의 환경 한 가지 실천 DAY",
        "challengeInfo": "",
        "expected": False
    },
    { # 57
        "verificationId": 1010,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/10_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 10,
        "challengeName": "양치컵 하나로 지구를 살려요!",
        "challengeInfo": "",
        "expected": True
    },
    { # 58
        "verificationId": 1020,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/10_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 10,
        "challengeName": "양치컵 하나로 지구를 살려요!",
        "challengeInfo": "",
        "expected": False
    },
    { # 59
        "verificationId": 1110,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/11_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 11,
        "challengeName": "호랑이를 지켜요! 숲을 위한 하루",
        "challengeInfo": "",
        "expected": True
    },
    { # 60
        "verificationId": 1120,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/11_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 11,
        "challengeName": "호랑이를 지켜요! 숲을 위한 하루",
        "challengeInfo": "",
        "expected": False
    },
    { # 61
        "verificationId": 1210,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/12_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 12,
        "challengeName": "꺼주세요 1시간! 에너지를 아끼는 시간 OFF",
        "challengeInfo": "",
        "expected": True
    },
    { # 62
        "verificationId": 1220,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/12_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 12,
        "challengeName": "꺼주세요 1시간! 에너지를 아끼는 시간 OFF",
        "challengeInfo": "",
        "expected": False
    },
    { # 63
        "verificationId": 1310,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/13_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 13,
        "challengeName": "버리지 마세요! 오늘은 자원순환 챌린지",
        "challengeInfo": "",
        "expected": True
    },
    { # 64
        "verificationId": 1320,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/13_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 13,
        "challengeName": "버리지 마세요! 오늘은 자원순환 챌린지",
        "challengeInfo": "",
        "expected": False
    },
    { # 65
        "verificationId": 1410,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/14_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 14,
        "challengeName": "오늘은 걷거나 타세요! Car-Free 실천 챌린지",
        "challengeInfo": "",
        "expected": True
    },
    { # 66
        "verificationId": 1420,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/14_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 14,
        "challengeName": "오늘은 걷거나 타세요! Car-Free 실천 챌린지",
        "challengeInfo": "",
        "expected": False
    },
    { # 67
        "verificationId": 1510,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/15_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 15,
        "challengeName": "기후재난 이야기 공유 챌린지",
        "challengeInfo": "",
        "expected": True
    },
    { # 68
        "verificationId": 1520,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/15_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 15,
        "challengeName": "기후재난 이야기 공유 챌린지",
        "challengeInfo": "",
        "expected": False
    },
    { # 69
        "verificationId": 1610,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/16_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 16,
        "challengeName": "오늘은 비건 한 끼, 지구와 나를 위한 식사",
        "challengeInfo": "",
        "expected": True
    },
    { # 70
        "verificationId": 1620,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/16_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 16,
        "challengeName": "오늘은 비건 한 끼, 지구와 나를 위한 식사",
        "challengeInfo": "",
        "expected": False
    },
    { # 71
        "verificationId": 210,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/2_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 2,
        "challengeName": "해양 정화로 고래를 지켜요",
        "challengeInfo": "",
        "expected": True
    },
    { # 72
        "verificationId": 220,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/2_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 2,
        "challengeName": "해양 정화로 고래를 지켜요",
        "challengeInfo": "",
        "expected": False
    },
    { # 73
        "verificationId": 610,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/6_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 6,
        "challengeName": "음식물도 순환돼요! 퇴비 챌린지",
        "challengeInfo": "",
        "expected": True
    },
    { # 74
        "verificationId": 620,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/6_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 6,
        "challengeName": "음식물도 순환돼요! 퇴비 챌린지",
        "challengeInfo": "",
        "expected": False
    },
    { # 75
        "verificationId": 1710,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/17_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 17,
        "challengeName": "한 뼘의 텃밭, 농민의 마음을 심어요",
        "challengeInfo": "",
        "expected": True
    },
    { # 76
        "verificationId": 1720,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/17_%E1%84%8B%E1%85%B5%E1%84%87%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 17,
        "challengeName": "한 뼘의 텃밭, 농민의 마음을 심어요",
        "challengeInfo": "",
        "expected": False
    },
    { # 77
        "verificationId": 9011,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/1_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 901,
        "challengeName": "텀블러 챌린지",
        "challengeInfo": "일회용 컵 대신 같이 각자 텀블러를 사용하는 습관을 길러요~!",
        "expected": True
    },
    { # 78
        "verificationId": 9012,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/1_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 901,
        "challengeName": "텀블러 챌린지",
        "challengeInfo": "일회용 컵 대신 같이 각자 텀블러를 사용하는 습관을 길러요~!",
        "expected": False
    },
    { # 79
        "verificationId": 9021,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/2_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.jpg",
        "challengeId": 902,
        "challengeName": "에코백 사용하기 챌린지",
        "challengeInfo": "가죽가방 대신 간편한 에코백 같이 사용해요~!",
        "expected": True
    },
    { # 80
        "verificationId": 9022,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/2_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 902,
        "challengeName": "에코백 사용하기 챌린지",
        "challengeInfo": "가죽가방 대신 간편한 에코백 같이 사용해요~!",
        "expected": False
    },
    { # 81
        "verificationId": 9031,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/3_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.jpg",
        "challengeId": 903,
        "challengeName": "장바구니 사용하기 챌린지",
        "challengeInfo": "비닐봉지 말고 장바구니를 사용해요!",
        "expected": True
    },
    { # 82
        "verificationId": 9032,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/3_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 903,
        "challengeName": "장바구니 사용하기 챌린지",
        "challengeInfo": "비닐봉지 말고 장바구니를 사용해요!",
        "expected": False
    },
    { # 83
        "verificationId": 9041,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/4_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.jpg",
        "challengeId": 904,
        "challengeName": "자전거 타기 챌린지",
        "challengeInfo": "자전거로 같이 출근해요~! 사진으로 인증해주세요.",
        "expected": True
    },
    { # 84
        "verificationId": 9042,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/4_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 904,
        "challengeName": "자전거 타기 챌린지",
        "challengeInfo": "자전거로 같이 출근해요~! 사진으로 인증해주세요.",
        "expected": False
    },
    { # 85
        "verificationId": 9051,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/5_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.jpg",
        "challengeId": 905,
        "challengeName": "대중교통 이용하기 챌린지",
        "challengeInfo": "버스나 지하철로 출퇴근해요.",
        "expected": True
    },
    { # 86
        "verificationId": 9052,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/5_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 905,
        "challengeName": "대중교통 이용하기 챌린지",
        "challengeInfo": "버스나 지하철로 출퇴근해요.",
        "expected": False
    },
    { # 87
        "verificationId": 9061,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/6_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.jpg",
        "challengeId": 906,
        "challengeName": "샐러드/채식 식단 먹기 챌린지",
        "challengeInfo": "고기 대신 샐러드나 채식 위주의 식사는 어떠신가요?",
        "expected": True
    },
    { # 88
        "verificationId": 9062,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/6_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 906,
        "challengeName": "샐러드/채식 식단 먹기 챌린지",
        "challengeInfo": "고기 대신 샐러드나 채식 위주의 식사는 어떠신가요?",
        "expected": False
    },
    { # 89
        "verificationId": 9071,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/7_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 907,
        "challengeName": "음식 남기지 않기 챌린지",
        "challengeInfo": "깨끗하게 비운 식판 또는 접시를 공유해요.",
        "expected": True
    },
    { # 90
        "verificationId": 9072,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/7_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 907,
        "challengeName": "음식 남기지 않기 챌린지",
        "challengeInfo": "깨끗하게 비운 식판 또는 접시를 공유해요.",
        "expected": False
    },
    { # 91
        "verificationId": 9081,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/8_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.jpg",
        "challengeId": 908,
        "challengeName": "계단 이용하기 챌린지",
        "challengeInfo": "낮은 층은 엘리베이터 대신 계단으로! 건강도 환경도 같이 챙겨요",
        "expected": True
    },
    { # 92
        "verificationId": 9082,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/8_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 908,
        "challengeName": "계단 이용하기 챌린지",
        "challengeInfo": "낮은 층은 엘리베이터 대신 계단으로! 건강도 환경도 같이 챙겨요",
        "expected": False
    },
    { # 93
        "verificationId": 9091,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/9_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 909,
        "challengeName": "재활용 분리수거 챌린지",
        "challengeInfo": "오늘 나온 쓰레기, 제대로 분리수거한 사진으로 인증해요.",
        "expected": True
    },
    { # 94
        "verificationId": 9092,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/9_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 909,
        "challengeName": "재활용 분리수거 챌린지",
        "challengeInfo": "오늘 나온 쓰레기, 제대로 분리수거한 사진으로 인증해요.",
        "expected": False
    },
    { # 95
        "verificationId": 9101,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/10_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 910,
        "challengeName": "손수건 사용 챌린지",
        "challengeInfo": "일회용 티슈 대신에 손수건을 사용해요",
        "expected": True
    },
    { # 96
        "verificationId": 9102,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/10_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 910,
        "challengeName": "손수건 사용 챌린지",
        "challengeInfo": "일회용 티슈 대신에 손수건을 사용해요",
        "expected": False
    },
    { # 97
        "verificationId": 9111,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/11_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.jpg",
        "challengeId": 911,
        "challengeName": "쓰레기 줍기 챌린지",
        "challengeInfo": "길에서 쓰레기 하나 줍는 것도 큰 실천이에요~!, 모범 시민이 되어보아요",
        "expected": True
    },
    { # 98
        "verificationId": 9112,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/11_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 911,
        "challengeName": "쓰레기 줍기 챌린지",
        "challengeInfo": "길에서 쓰레기 하나 줍는 것도 큰 실천이에요~!, 모범 시민이 되어보아요",
        "expected": False
    },
    { # 99
        "verificationId": 9121,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/12_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 912,
        "challengeName": "안쓰는 전기 플러그 뽑기 챌린지",
        "challengeInfo": "안 쓰는 플러그는 빼요! 전기 절약하고 나뭇잎 챙겨요~!",
        "expected": True
    },
    { # 100
        "verificationId": 9122,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/12_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 912,
        "challengeName": "안쓰는 전기 플러그 뽑기 챌린지",
        "challengeInfo": "안 쓰는 플러그는 빼요! 전기 절약하고 나뭇잎 챙겨요~!",
        "expected": False
    },
    { # 101
        "verificationId": 9131,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/13_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.jpg",
        "challengeId": 913,
        "challengeName": "고체 비누 사용하기 챌린지",
        "challengeInfo": "세숫비누 사용으로 옛날 감성 느껴요~!",
        "expected": True
    },
    { # 102
        "verificationId": 9132,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/13_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 913,
        "challengeName": "고체 비누 사용하기 챌린지",
        "challengeInfo": "세숫비누 사용으로 옛날 감성 느껴요~!",
        "expected": False
    },
    { # 103
        "verificationId": 9141,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/14_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 914,
        "challengeName": "하루 만보 걷기 챌린지",
        "challengeInfo": "오늘도 10,000보 성공..? 걸음 수 인증샷으로 같이 공유해요",
        "expected": True
    },
    { # 104
        "verificationId": 9142,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/14_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.jpg",
        "challengeId": 914,
        "challengeName": "하루 만보 걷기 챌린지",
        "challengeInfo": "오늘도 10,000보 성공..? 걸음 수 인증샷으로 같이 공유해요",
        "expected": False
    },
    { # 105
        "verificationId": 9151,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/15_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.jpg",
        "challengeId": 915,
        "challengeName": "도시락 싸먹기 챌린지",
        "challengeInfo": "일회용 대신 정성이 담긴 도시락으로 점심을 함께해요~!",
        "expected": True
    },
    { # 106
        "verificationId": 9152,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/15_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 915,
        "challengeName": "도시락 싸먹기 챌린지",
        "challengeInfo": "일회용 대신 정성이 담긴 도시락으로 점심을 함께해요~!",
        "expected": False
    },
    { # 107
        "verificationId": 9161,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/16_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.jpg",
        "challengeId": 916,
        "challengeName": "작은 텃밭 가꾸기 챌린지",
        "challengeInfo": "베란다든 화분이든, 내 손으로 키운 초록이들 공유해요.",
        "expected": True
    },
    { # 108
        "verificationId": 9162,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/16_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 916,
        "challengeName": "작은 텃밭 가꾸기 챌린지",
        "challengeInfo": "베란다든 화분이든, 내 손으로 키운 초록이들 공유해요.",
        "expected": False
    },
    { # 109
        "verificationId": 9171,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/17_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 917,
        "challengeName": "반려 식물 인증하기 챌린지",
        "challengeInfo": "내 책상 위 작은 식물, 같이 공유해요",
        "expected": True
    },
    { # 110
        "verificationId": 9172,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/17_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 917,
        "challengeName": "반려 식물 인증하기 챌린지",
        "challengeInfo": "내 책상 위 작은 식물, 같이 공유해요",
        "expected": False
    },
    { # 111
        "verificationId": 9181,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/18_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.jpg",
        "challengeId": 918,
        "challengeName": "전자 영수증 받기 챌린지",
        "challengeInfo": "종이대신 스마트하게! 전자 영수증 받은 화면을 인증해요~!",
        "expected": True
    },
    { # 112
        "verificationId": 9182,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/18_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 918,
        "challengeName": "전자 영수증 받기 챌린지",
        "challengeInfo": "종이대신 스마트하게! 전자 영수증 받은 화면을 인증해요~!",
        "expected": False
    },
    { # 113
        "verificationId": 9191,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/19_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 919,
        "challengeName": "친환경 인증 마크 상품 구매하기 챌린지",
        "challengeInfo": "환경을 생각한 소비! 친환경 마크가 찍힌 제품을 인증해요",
        "expected": True
    },
    { # 114
        "verificationId": 9192,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/19_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 919,
        "challengeName": "친환경 인증 마크 상품 구매하기 챌린지",
        "challengeInfo": "환경을 생각한 소비! 친환경 마크가 찍힌 제품을 인증해요",
        "expected": False
    },
    { # 115
        "verificationId": 9201,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/20_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.jpg",
        "challengeId": 920,
        "challengeName": "다회용기 사용하기 챌린지",
        "challengeInfo": "일회용품은 이제 그만! 다회용기에 담긴 음식이나 음료를 인증해주세요",
        "expected": True
    },
    { # 116
        "verificationId": 9202,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/20_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 920,
        "challengeName": "다회용기 사용하기 챌린지",
        "challengeInfo": "일회용품은 이제 그만! 다회용기에 담긴 음식이나 음료를 인증해주세요",
        "expected": False
    },
    { # 117
        "verificationId": 9211,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/21_%E1%84%89%E1%85%A5%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC.png",
        "challengeId": 921,
        "challengeName": "대나무 칫솔 사용하기 챌린지",
        "challengeInfo": "플라스틱 대신 자연으로~~ 대나무 칫솔로 하루와 끝을 마무리해요",
        "expected": True
    },
    { # 118
        "verificationId": 9212,
        "type": "GROUP",
        "imageUrl": "https://storage.googleapis.com/leafresh-images/init/21_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%91%E1%85%A2.png",
        "challengeId": 921,
        "challengeName": "대나무 칫솔 사용하기 챌린지",
        "challengeInfo": "플라스틱 대신 자연으로~~ 대나무 칫솔로 하루와 끝을 마무리해요",
        "expected": False
    }
]




















