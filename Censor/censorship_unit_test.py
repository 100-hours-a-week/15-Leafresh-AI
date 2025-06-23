from model.censorship_hyperclovax_model import HyperClovaxModel
from model.LLM_censorship_model import CensorshipModel
import time
from censorship_test_case import test_cases
import pandas as pd
from tabulate import tabulate
from wcwidth import wcswidth

# model = HyperClovaxModel()
model = CensorshipModel()
results = []
response_times = []

def tabulate_fixed(df: pd.DataFrame):
    """한글 너비 고려한 tabulate 정렬"""
    df_fixed = df.copy()
    col_widths = {
        col: max(wcswidth(str(val)) for val in df[col]) + 2
        for col in df.columns
    }
    for col in df.columns:
        df_fixed[col] = df_fixed[col].apply(lambda x: str(x).ljust(col_widths[col] - wcswidth(str(x)) + len(str(x))))

    return tabulate(df_fixed, headers="keys", tablefmt="pretty", showindex=False)

count = 0
for idx, case in enumerate(test_cases, 1):
    start_time = time.time()

    if not case.get("challenge"):
        result, message = True, "챌린지 목록이 없으므로 생성 가능합니다."
    else:
        result, message = model.validate(case["challengeName"], case["startDate"], case["endDate"], case["challenge"])

    end_time = time.time()
    duration = round(end_time - start_time, 3)  # 소숫점 3자리로 반올림
    response_times.append(duration)

    is_pass = result == case["expected"]
 
    results.append({
        "Test #": idx,
        "Challenge Name": case["challengeName"],
        "Expected": case["expected"],
        "Actual": result,
        "Pass": is_pass,
        "Response Time (s)": duration
    })

    if is_pass:
        print(f"[Test {idx}] ✅ ({duration}s)")
    else:
        print(f"[Test {idx}] ❗ ({duration}s)")

# 보고서 요약 출력
df = pd.DataFrame(results)
total = len(df)
passed = df["Pass"].sum()
accuracy = round((passed / total) * 100, 2)

false_to_true = df[(df["Expected"] == False) & (df["Actual"] == True)]
true_to_false = df[(df["Expected"] == True) & (df["Actual"] == False)]

false_to_true_rate = round((len(false_to_true) / total) * 100, 2)
true_to_false_rate = round((len(true_to_false) / total) * 100, 2)

# 평균 응답 시간 정리
avg_time = round(sum(response_times) / len(response_times), 3)

# 실패한 테스트 목록 정리
failed_df = df[df["Pass"] == False][["Test #", "Challenge Name", "Expected", "Actual"]]

print("\n" + "=" * 60)
print("LLM Censorship Model 테스트 보고서 요약")
print("=" * 60)
print(f"  - 총 테스트 수: {total}")
print(f"  - 통과한 테스트 수: {passed}")
print(f"  - 테스트 정확도: {accuracy}%")
print(f"  - False → True (오탐): {len(false_to_true)}개 ({false_to_true_rate}%)")
print(f"  - True → False (누락): {len(true_to_false)}개 ({true_to_false_rate}%)")
print(f"  - 평균 응답 시간: {avg_time}s")
print("=" * 60)

if not failed_df.empty:
    print("\n[ 실패한 테스트 목록 ]\n")
    print(tabulate_fixed(failed_df) + "\n")
else:
    print("\n모든 테스트가 통과되었습니다! \n")

# 테스트 종료 후 대기 
time.sleep(2)
