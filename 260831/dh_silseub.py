import numpy as np
import pandas as pd

df = pd.read_csv("설비배치1.csv", encoding="utf-8-sig")
sensor = ["온도", "진동", "회전수", "압력"]

print(df.shape)
dic = {}
dic["온도"] = df["온도"].isnull().sum().item()
dic["압력"] = df["압력"].isnull().sum().item()
print(dic)
print(type(df["진동"][1]).__name__)
dic1 = {}
dic1["정상"] = sum(df["판정"] == "정상")
dic1["주의"] = sum(df["판정"] == "주의")
dic1["이상"] = sum(df["판정"] == "이상")

print(dic1)

# 2
# for i in range(len(df["진동"])):
#     try:
#         df["진동"][i] = float(df["진동"][i])
#     except ValueError or TypeError:
#         df["진동"][i] = None
# print(df["진동"])

df["진동"] = pd.to_numeric(df["진동"], errors="coerce")

print(df["진동"].isnull().sum())
print(df["진동"].mean().round(2))

# 3
print(df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)
print(df.shape)

# 4
a = df["온도"].mean()
b = df["압력"].median()
c = df["진동"].mean()
df["온도"] = df["온도"].fillna(a)
df["압력"] = df["압력"].fillna(b)
df["진동"] = df["진동"].fillna(c)

sum = 0
for i in sensor:
    sum += df[i].isnull().sum()
print(sum)
print(a.round(2), b.round(2))

# 5
result = df.groupby("생산라인")[["온도", "진동", "회전수", "압력"]].mean().round(2)
print(result)
print(df.groupby("생산라인")["검사일시"].count().to_dict())

# 6
avg = df["온도"].mean()
std = df["온도"].std(ddof=0)

print(avg.round(2), std.round(2))
z = (df["온도"] - avg) / std

print((abs(z) > 3).sum(), (abs(z) > 2).sum())

# 7
q1 = np.percentile(df["압력"], 25)
q3 = np.percentile(df["압력"], 75)
iqr = q3 - q1
low = q1 - iqr * 1.5
high = q3 + iqr * 1.5
print(low.round(2), high.round(2))
print(((df["압력"] < low) | (df["압력"] > high)).sum())
print(
    df[(df["압력"] < low) | (df["압력"] > high)]
    .groupby("생산라인")["검사일시"]
    .count()
    .to_dict()
)

# 8

print(df.groupby("생산라인")["검사일시"].count().to_dict())
df.drop(df[(df["압력"] < low) | (df["압력"] > high)].index, inplace=True)
print(df.groupby("생산라인")["검사일시"].count().to_dict())
print(df.shape)

# 9

for i in sensor:
    mn = df[i].min()
    mx = df[i].max()
    z = (df[i] - mn) / (mx - mn)
    df[f"z_{i}"] = z

print(df[sensor].min().to_dict())
print(df[sensor].max().to_dict())
print(df[sensor].mean().round(3).to_dict())
print(df[["검사일시", "생산라인", "온도", "진동", "회전수", "압력"]].shape)
df[["검사일시", "생산라인", "온도", "진동", "회전수", "압력"]].to_csv(
    "정규화_멘티.csv ", index=False, encoding="utf-8-sig"
)

# df = df.copy()
# if 기준 is None:
#     # 기준을 안 받았으면 지금 데이터로 min·max 를 만든다 (= 학습 데이터일 때)
#     기준 = {c: {"min": float(df[c].min()), "max": float(df[c].max())} for c in cols}
# # 받았거나 방금 만든 기준을 각 열에 적용: (값-최소)/(최대-최소)
# for c in cols:
#     lo = 기준[c]["min"]
#     hi = 기준[c]["max"]
#     df[c] = ((df[c] - lo) / (hi - lo)).round(3)
# return df, 기준

# 10
dic = {"A라인": 0, "B라인": 1, "C라인": 2}
df["라인코드"] = df["생산라인"].map(dic)

df[
    ["검사일시", "생산라인", "라인코드", "온도", "진동", "회전수", "압력", "판정"]
].to_csv("정제결과_멘티.csv", index=False, encoding="utf-8-sig")

df1 = pd.read_csv("정제결과_멘티.csv")
print(df1.shape, df1.isnull().sum().sum(), df1.duplicated().sum().sum())

print(df1.columns.to_list())
