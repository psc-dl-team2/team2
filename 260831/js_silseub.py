import numpy as np
import pandas as pd

df = pd.read_csv("설비배치1.csv", encoding="utf-8-sig")
mte = pd.read_csv("정제결과_멘티.csv", encoding="utf-8-sig")
mte_norm = pd.read_csv("정규화_멘티.csv", encoding="utf-8-sig")
sensor = ["온도", "진동", "회전수", "압력"]
keys = ["검사일시", "생산라인", "설비번호"]

print("\n ===== 문제 1 =====")
print(len(df), len(df.drop_duplicates()), len(mte))
print(df.groupby("생산라인")["온도"].size())
print(mte.groupby("생산라인")["온도"].size())
df_groupby_temp = df.groupby("생산라인")["온도"].mean()
mte_groupby_temp = mte.groupby("생산라인")["온도"].mean()
a = pd.DataFrame(
    {
        "원본": df_groupby_temp.round(2),
        "멘티": mte_groupby_temp.round(2),
        "차이": (df_groupby_temp - mte_groupby_temp).round(2),
    }
)
print(a)
"""
C라인만 4개의 행이 사라졌다. 그 이유는 중복 제거.
출력 첫번째에서 중복 제거 전 - 186개의 행, 중복 제거 후 - 182개의 행

A라인만 0.6도가 상승했다.

라인별 공정성 문제가 생길 수 있다.
>> ai 추가 답변 : 표본 편향, 센서나 통신 장애 가능성
"""

print("\n ===== 문제 2 =====")
print(mte.duplicated(subset=["검사일시", "생산라인", "라인코드"]).sum())
print(
    mte[mte.duplicated(subset=["검사일시", "생산라인", "라인코드"])][
        ["검사일시", "생산라인", "라인코드", "온도", "압력"]
    ]
)
mto = mte.drop_duplicates(subset=["검사일시", "생산라인", "라인코드"]).reset_index(
    drop=True
)
print(mto.shape, mto.groupby("생산라인")["검사일시"].count().to_dict())
mte_gb_ = mte.groupby("생산라인")[["온도", "압력"]].mean()
mto_gb_ = mto.groupby("생산라인")[["온도", "압력"]].mean()
print((mto_gb_ - mte_gb_).round(2))
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
print(mto["온도"][z > 2.5].to_list())
avg_gb = mto.groupby("생산라인")["온도"].mean().round(5)
std_gb = mto.groupby("생산라인")["온도"].std().round(5)
avg_gb_dic = avg_gb.to_dict()
std_gb_dic = std_gb.to_dict()
z_ = (mto["온도"] - mto["생산라인"].map(avg_gb_dic)) / mto["생산라인"].map(std_gb_dic)
print(mto["온도"][z_ > 2.5].to_list())
print(
    mto[["검사일시", "생산라인", "라인코드", "온도"]].iloc[z_.index[z_ > 2.5].to_list()]
)
"""
생산라인별로 기계가 다르기 때문에 같은 데이터라고 생각하면 안 된다.
그러므로 각각의 데이터에 대한 표준화를 진행해야 옳은 판단을 할 수 있는 데이터가 마련된다.
"""

print("\n ===== 문제 4 =====")
mto_gb_temp = mto.groupby("생산라인")["온도"].mean()
print(mto_gb_temp)
mto_remove_outlier = mto[z_ <= 2.5]
mto_ro_gb_temp = mto_remove_outlier.groupby("생산라인")["온도"].mean()
print(mto_ro_gb_temp)
print(mto_gb_temp - mto_ro_gb_temp)
df["온도"] = df["온도"].fillna(df["생산라인"].map(mto_ro_gb_temp.to_dict()))
df["압력"] = df["압력"].fillna(
    df["생산라인"].map(df.groupby("생산라인")["압력"].median())
)
df["진동"] = pd.to_numeric(df["진동"], errors="coerce")
df["진동"] = df["진동"].fillna(
    df["생산라인"].map(df.groupby("생산라인")["압력"].median())
)
print(df.isnull().sum())
print(df.groupby("생산라인")["온도"].mean())
print(mte.groupby("생산라인")["온도"].mean())
"""
이상값을 제외한 평균을 사용하는 것이 맞다.
이상값도 값이라고 생각할 수 있지만, z점수 임계값 2.5는 신뢰수준 98.76%이다.
즉, 이상치가 전체 값의 상위 또는 하위 0.62%에 해당하기에 제외해도 차질이 없다.

중앙값을 사용하면 이상치의 특성이 포함되지 않기에 위와 같은 문제가 발생할 확률이 적다.
"""
