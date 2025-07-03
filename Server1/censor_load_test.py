import asyncio
import aiohttp
import time
import statistics

# 테스트 조건
TOTAL_REQUESTS = 1000
CONCURRENT_TASKS = 2
URL = 'http://localhost:7000/ai/challenges/group/validation'

# 응답 시간 저장 리스트
response_times = []

# 요청 payload 정의
def create_payload():
    return {
        "memberId": 1,
        "challengeName": "제로웨이스트 실천하기",
        "startDate": "2025-07-01",
        "endDate": "2025-07-31",
        "challenge": [
            {
                "id": 1,
                "name": "플라스틱 줄이기",
                "startDate": "2025-07-01",
                "endDate": "2025-07-10"
            },
            {
                "id": 2,
                "name": "텀블러 사용하기",
                "startDate": "2025-07-15",
                "endDate": "2025-07-25"
            }
        ]
    }

# 개별 요청 함수
async def send_request(session, semaphore, request_id):
    async with semaphore:
        start = time.perf_counter()
        try:
            async with session.post(URL, json=create_payload()) as response:
                end = time.perf_counter()
                response_times.append(end - start)
                if response.status == 200:
                    print(f"[INFO] 요청 {request_id} 완료 (응답 시간: {end - start:.2f}초)")
                else:
                    print(f"[WARN] 요청 {request_id} 실패 (상태 코드: {response.status})")
        except Exception as e:
            print(f"[ERROR] 요청 {request_id} 실패: {e}")


# 전체 실행 함수
async def run_load_test():
    semaphore = asyncio.Semaphore(CONCURRENT_TASKS)
    async with aiohttp.ClientSession() as session:
        tasks = [
            asyncio.create_task(send_request(session, semaphore, i + 1))
            for i in range(TOTAL_REQUESTS)
        ]
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    start_time = time.time()
    asyncio.run(run_load_test())
    elapsed_time = time.time() - start_time

    # 통계 출력
    if response_times:
        avg = statistics.mean(response_times) 
        minimum = min(response_times)
        maximum = max(response_times)
        print(f"\n총 {TOTAL_REQUESTS}회 요청 완료 (총 소요 시간: {elapsed_time:.2f}초)")
        print(f"평균 응답 시간: {avg:.2f}초")
        print(f"가장 짧은 응답 시간: {minimum:.2f}초")
        print(f"가장 긴 응답 시간: {maximum:.2f}초")
    else:
        print("요청이 모두 실패했습니다.")
