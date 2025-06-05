from model.verify.LLM_verify_model import ImageVerifyModel
from verify_test_case import test_cases
import pandas as pd
from tabulate import tabulate
from wcwidth import wcswidth
from urllib.parse import unquote
import pillow_avif
import time

model = ImageVerifyModel()
results = []

BUCKET_NAME = "leafresh-images"

def extract_blob_name(image_url: str) -> str:
    """imageUrl에서 blob 이름 추출 (예: init/example.png)"""
    parts = image_url.split("/")
    init_index = parts.index("init")
    blob_path = "/".join(parts[init_index:])
    return unquote(blob_path)

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
# for idx, case in enumerate(test_cases[63:64], start=64):
    try:
        blob_name = extract_blob_name(case["imageUrl"])
        output = model.image_verify(BUCKET_NAME, blob_name, case["type"], case["challengeId"], case["challengeName"])
        result_text = output.strip()
        result = result_text.startswith("예")
        
        is_pass = result == case["expected"]

        results.append({
            "Test #": idx,
            "verificationId": case["verificationId"],
            "challengeName": case["challengeName"], 
            "Expected": case["expected"],
            "Actual": result,
            "Pass": is_pass,
            # "LLM Output": output.strip()
        })

        if is_pass:
            print(f"[Test {idx}]")
        else:
            print(f"[Test {idx}] ❗")

    except Exception as e:
        print(f"[Test {idx}] 에러 발생: {e}")


# 보고서 요약 출력
df = pd.DataFrame(results)
total = len(df)
passed = df["Pass"].sum()
accuracy = round((passed / total) * 100, 2)

false_to_true = df[(df["Expected"] == False) & (df["Actual"] == True)]
true_to_false = df[(df["Expected"] == True) & (df["Actual"] == False)]

false_to_true_rate = round((len(false_to_true) / total) * 100, 2)
true_to_false_rate = round((len(true_to_false) / total) * 100, 2)

# 실패한 테스트 목록 정리
failed_df = df[df["Pass"] == False][["Test #", "verificationId", "challengeName", "Expected", "Actual"]] #, "LLM Output"]]

print("\n" + "=" * 60)
print("ImageVerifyModel 테스트 보고서 요약")
print("=" * 60)
print(f"  - 총 테스트 수: {total}")
print(f"  - 통과한 테스트 수: {passed}")
print(f"  - 테스트 정확도: {accuracy}%")
print(f"  - False → True (오탐): {len(false_to_true)}개 ({false_to_true_rate}%)")
print(f"  - True → False (누락): {len(true_to_false)}개 ({true_to_false_rate}%)")
print("=" * 60)

if not failed_df.empty:
    print("\n[ 실패한 테스트 목록 ]\n")
    print(tabulate_fixed(failed_df) + "\n")
else:
    print("\n모든 테스트가 통과되었습니다! \n")

time.sleep(2)

