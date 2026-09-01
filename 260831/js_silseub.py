import numpy as np
import pandas as pd

df = pd.read_csv("설비배치1.csv", encoding="utf-8-sig")
mte = pd.read_csv("정제결과_멘티.csv", encoding="utf-8-sig")
mte_norm = pd.read_csv("정규화_멘티.csv", encoding="utf-8-sig")
df2 = pd.read_csv("설비배치2.csv", encoding="utf-8-sig")
sensor = ["온도", "진동", "회전수", "압력"]
keys = ["검사일시", "생산라인", "설비번호"]
mte.rename(columns={"라인코드": "설비번호"}, inplace=True)

print("\n ===== 문제 1 =====")
print("[세 표의 행 수]")
print(
    "원본 : ",
    len(df),
    ", 중복 제거 : ",
    len(df.drop_duplicates(subset=keys)),
    ", 멘티 : ",
    len(mte),
)
print("\n[생산라인별 행 수]")
df_dup = df.drop_duplicates(subset=keys).reset_index(drop=True)
print(df_dup.groupby("생산라인")["온도"].size())
print(mte.groupby("생산라인")["온도"].size())
df_groupby_temp = df_dup.groupby("생산라인")["온도"].mean()
mte_groupby_temp = mte.groupby("생산라인")["온도"].mean()
print("\n[생산라인별 온도 평균] - 원본 / 멘티 / 차이")
a = pd.DataFrame(
    {
        "원본": df_groupby_temp.round(2),
        "멘티": mte_groupby_temp.round(2),
        "차이": (df_groupby_temp - mte_groupby_temp).round(2),
    }
)
print(a)
"""
C라인만 3개의 행이 사라졌다. 그 이유는 중복 제거.
출력 첫번째에서 중복 제거 전 - 186개의 행, 중복 제거 후 - 180개의 행

A라인만 0.55도가 상승했다.

라인별 공정성 문제가 생길 수 있다.
>> ai 추가 답변 : 표본 편향, 센서나 통신 장애 가능성
"""

print("\n ===== 문제 2 =====")
print("[keys별 중복 쌍]", mte.duplicated(subset=keys).sum(), "개")
print()
print(
    mte[mte.duplicated(subset=keys)][
        ["검사일시", "생산라인", "설비번호", "온도", "압력"]
    ]
)
mto = mte.drop_duplicates(subset=keys).reset_index(drop=True)
print(
    "\n[표 크기]",
    mto.shape,
    " / [라인별 행 수]",
    mto.groupby("생산라인")["검사일시"].count().to_dict(),
)
mte_gb_ = mte.groupby("생산라인")[["온도", "압력"]].mean()
mto_gb_ = mto.groupby("생산라인")[["온도", "압력"]].mean()
print(f"[행 그대로 평균]\n{mte_gb_.round(3)}")
print(f"[중복 제거 평균]\n{mto_gb_.round(3)}")
# print((mto_gb_ - mte_gb_).round(2))
"""
제거 전과 후의 차이는 온도에서 나타난다. B라인에 0.04의 차이가 있다.
0.04면 큰 차이는 없어보이나, 정밀 가공 등 유의미한 변화가 일어날 수 있다.
drop_duplicates()로 못 잡은 이유는 행 전체가 같아야 제거되기 때문이다.
센서 값의 미세한 차이를 다르다고 인식하여 제거되지 못했다.
현실에서도 센서 및 통신 장애로 생길 중복값들을 주의하자.
"""
print("\n ===== 문제 3 =====")
avg = mto["온도"].mean()
std = mto["온도"].std(ddof=0)
z = (mto["온도"] - avg) / std
print("(가) 기준", len(mto["온도"][z > 2.5].to_list()))

avg_gb = mto.groupby("생산라인")["온도"].mean().round(5)
std_gb = mto.groupby("생산라인")["온도"].std(ddof=0).round(5)
avg_gb_dic = avg_gb.to_dict()
std_gb_dic = std_gb.to_dict()
z_ = (mto["온도"] - mto["생산라인"].map(avg_gb_dic)) / mto["생산라인"].map(std_gb_dic)

print("(나) 기준", len(mto["온도"][abs(z_) > 2.5].to_list()))
print(mto.loc[abs(z_) > 2.5, ["검사일시", "생산라인", "설비번호", "온도"]])
print(avg_gb_dic)
"""
생산라인별로 기계가 다르기 때문에 같은 데이터라고 생각하면 안 된다.
그러므로 각각의 데이터에 대한 표준화를 진행해야 옳은 판단을 할 수 있는 데이터가 마련된다.
"""

print("\n ===== 문제 4 =====")
mto_gb_temp = mto.groupby("생산라인")["온도"].mean()
print("[이상값 포함]", mto_gb_temp)
mto_remove_outlier = mto[z_.abs() <= 2.5]
mto_ro_gb_temp = mto_remove_outlier.groupby("생산라인")["온도"].mean()
print("[이상값 제외]", mto_ro_gb_temp)
print("[두 값의 차이]", mto_gb_temp - mto_ro_gb_temp)

df_dup["온도"] = df_dup["온도"].fillna(df_dup["생산라인"].map(mto_ro_gb_temp))
df_dup["압력"] = df_dup["압력"].fillna(
    df_dup["생산라인"].map(df_dup.groupby("생산라인")["압력"].median())
)
df_dup["진동"] = pd.to_numeric(df_dup["진동"], errors="coerce")
df_dup["진동"] = df_dup["진동"].fillna(
    df_dup["생산라인"].map(df_dup.groupby("생산라인")["진동"].median())
)

# df["온도"] = df["온도"].fillna(df["생산라인"].map(mto_ro_gb_temp))
# df["진동"] = pd.to_numeric(df["진동"], errors="coerce")
# df["진동"] = df["진동"].fillna(
#     df["생산라인"].map(df.groupby("생산라인")["진동"].median())
# )
# df["압력"] = df["압력"].fillna(
#     df["생산라인"].map(df.groupby("생산라인")["압력"].median())
# )

# print("결측치 개수:\n", df.isnull().sum())

print("[남은 결측 총 개수]", df_dup.isnull().sum().sum())
print("[라인별 온도 평균]", df_dup.groupby("생산라인")["온도"].mean())
print("[멘티의 라인별 온도 평균]", mto.groupby("생산라인")["온도"].mean())
"""
이상값을 제외한 평균을 사용하는 것이 맞다.
이상값도 값이라고 생각할 수 있지만, z점수 임계값 2.5는 신뢰수준 98.76%이다.
즉, 이상치가 전체 값의 상위 또는 하위 0.62%에 해당하기에 제외해도 차질이 없다.

중앙값을 사용하면 이상치의 특성이 포함되지 않기에 위와 같은 문제가 발생할 확률이 적다.
"""


print("\n ===== 문제 5 =====")
# print(df.groupby("생산라인")["압력"].count())
press_c = df_dup[df_dup["생산라인"] == "C라인"]["압력"]
press_c_std = press_c.std(ddof=0)
print(press_c_std)

# 1
press_gb_avg = df_dup.groupby("생산라인")["압력"].mean().to_dict()
press_gb_std = df_dup.groupby("생산라인")["압력"].std(ddof=0).to_dict()
press_z = (df_dup["압력"] - df_dup["생산라인"].map(press_gb_avg)) / df_dup[
    "생산라인"
].map(press_gb_std)

press_over3 = press_z[press_z > 3]
print(
    len(press_over3),
    df_dup.iloc[press_over3.index.to_list()][["생산라인", "압력"]],
)
press_gb_med = df_dup.groupby("생산라인")["압력"].median().to_dict()
df_dup.loc[press_over3.index, "압력"] = df_dup.loc[press_over3.index, "생산라인"].map(
    press_gb_med
)
print(df_dup.loc[press_over3.index, ["생산라인", "압력"]])

press_c_z = (press_c - press_c.mean()) / press_c_std
# print(press_c_z)

# press_c = df[df["생산라인"] == "C라인"]["압력"]
# press_c_std = press_c.std()
# print(press_c_std)

# 2
press_gb_avg = df_dup.groupby("생산라인")["압력"].mean().to_dict()
press_gb_std = df_dup.groupby("생산라인")["압력"].std(ddof=0).to_dict()
press_z = (df_dup["압력"] - df_dup["생산라인"].map(press_gb_avg)) / df_dup[
    "생산라인"
].map(press_gb_std)
press_over3 = press_z[press_z > 3]
print(
    len(press_over3),
    df_dup.iloc[press_over3.index.to_list()][["생산라인", "압력"]],
)
press_gb_med = df_dup.groupby("생산라인")["압력"].median().to_dict()
df_dup.loc[press_over3.index, "압력"] = df_dup.loc[press_over3.index, "생산라인"].map(
    press_gb_med
)
print(df_dup.loc[press_over3.index, ["생산라인", "압력"]])

# 3
press_gb_avg = df_dup.groupby("생산라인")["압력"].mean().to_dict()
press_gb_std = df_dup.groupby("생산라인")["압력"].std(ddof=0).to_dict()
press_z = (df_dup["압력"] - df_dup["생산라인"].map(press_gb_avg)) / df_dup[
    "생산라인"
].map(press_gb_std)
press_over3 = press_z[press_z > 3]
print(
    len(press_over3),
    df_dup.iloc[press_over3.index.to_list()][["생산라인", "압력"]],
)

print(df_dup.groupby("생산라인")["압력"].count())
"""
평균과 표준편차는 이상치의 값도 포함되어 이상치에 민감하다.
두번째 필터링에서 새로운 값이 생긴 이유는 이전 이상치를 중앙값으로 수정하여,
필털핑 범위 축소에 있다.
세번째 필터링에서 걸러지지 않은 이유는 
하지만 z점수 필터링을 여러 번하는 게 좋은가?
>> 과적합 발생 가능성
"""

print("\n ===== 문제 6 =====")
for i in sensor:
    mn = df_dup.groupby("생산라인")[i].transform("min")
    mx = df_dup.groupby("생산라인")[i].transform("max")
    df_dup[f"nor_{i}"] = (df_dup[i] - mn) / (mx - mn)

mto_norm = df_dup[
    ["검사일시", "생산라인", "nor_온도", "nor_진동", "nor_회전수", "nor_압력"]
]

print("[맞춰진 행 개수]", len(mto_norm), len(df_dup))
mte_norm = mte_norm.sort_values(["검사일시", "생산라인"]).reset_index(drop=True)
mto_norm = mto_norm.sort_values(["검사일시", "생산라인"]).reset_index(drop=True)


for i in sensor:
    # diff = abs(mte_norm[i] - mto_norm[f"nor_{i}"])
    # over_005_count = (diff > 0.05).sum()
    # max_diff = diff.max()
    diff = abs(mte_norm[i] - mto_norm[f"nor_{i}"])
    diff_mx = diff.max()
    print(
        i, "차이 최대값 : ", diff_mx.round(3), "/ 0.05 이상", (diff > 0.05).sum(), "개"
    )
    if i == "온도":
        lst = diff.dropna().sort_values(ascending=False).index[:4].to_list()

t4_df = pd.DataFrame(
    {
        "검사일시": mto_norm.loc[lst, "검사일시"],
        "생산라인": mto_norm.loc[lst, "생산라인"],
        "멘티값": mte_norm.loc[lst, "온도"],
        "멘토값": mto_norm.loc[lst, "nor_온도"],
    }
)
print(t4_df)

mte_norm_avg = mte_norm.groupby("생산라인")["온도"].mean()
mto_norm_avg = mto_norm.groupby("생산라인")["nor_온도"].mean()

print(pd.concat([mte_norm_avg, mto_norm_avg], axis=1))
