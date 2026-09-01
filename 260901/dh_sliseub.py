import numpy as np
import pandas as pd

df = pd.read_csv("로그배치1.csv", encoding="utf-8-sig")
지표 = ["CPU온도", "전력", "응답시간", "메모리"]

# 1
print(df.shape)
df1 = df.isnull().sum()
print(df1[df1 > 0].to_dict())
print(type(df["전력"][0]).__name__)
print(df.groupby("상태")["수집시각"].count().to_dict())

# 2
df["전력"] = pd.to_numeric(df["전력"], errors="coerce")
print(df["전력"].isnull().sum())
print(df["전력"].mean().round(2))

# 3
print(df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)
print(df.shape)

# 4
a = df["CPU온도"].mean()
b = df["메모리"].median()
c = df["전력"].mean()

df["CPU온도"] = df["CPU온도"].fillna(a)
df["메모리"] = df["메모리"].fillna(b)
df["전력"] = df["전력"].fillna(c)

sum = 0
for i in 지표:
    sum += df[i].isnull().sum()

print(sum)
print(a.round(2), b.round(2))

# 5
df1 = df.groupby("구역")[["CPU온도", "전력", "응답시간", "메모리"]].mean().round(2)
print(df1)
print(df.groupby("구역")["수집시각"].count().to_dict())

# 6
m = df["CPU온도"].mean()
s = df["CPU온도"].std(ddof=0)
z = (df["CPU온도"] - m) / s
print(m.round(2), s.round(2))
print(int(df["CPU온도"][(abs(z) > 3)].sum()), int(df["CPU온도"][(abs(z) > 2)].sum()))

# 7
q1 = np.percentile(df["메모리"], 25)
q3 = np.percentile(df["메모리"], 75)
iqr = q3 - q1
low = q1 - iqr * 1.5
high = q3 + iqr * 1.5
print(low.round(2), high.round(2))
i = (df["메모리"] < low) | (df["메모리"] > high)
print(i.sum())
print(df.loc[i, "구역"].value_counts().to_dict())

# 8
print(df.groupby("구역")["수집시각"].count().to_dict())
df = df.drop(df[(df["메모리"] < low) | (df["메모리"] > high)].index).reset_index(
    drop=True
)
print(df.groupby("구역")["수집시각"].count().to_dict())
print(df.shape)

# 9
df1 = df[["수집시각", "구역"]].copy()
for i in 지표:
    mn = df[i].min()
    mx = df[i].max()
    z = (df[i] - mn) / (mx - mn)
    df1[i] = z

print(df1[지표].min().to_dict())
print(df1[지표].max().to_dict())
print(df1[지표].mean().round(3).to_dict())

df1.to_csv("정규화_멘티.csv", index=False, encoding="utf-8-sig")
df2 = pd.read_csv("정규화_멘티.csv", encoding="utf-8-sig")
print(df2.shape)

# 10
dic = {"Z1-알파": 0, "Z2-브라보": 1, "Z3-찰리": 2}
df["구역코드"] = df["구역"].map(dic)
df3 = df[
    ["수집시각", "구역", "구역코드", "CPU온도", "전력", "응답시간", "메모리", "상태"]
]
df3.to_csv("정제결과_멘티.csv", index=False, encoding="utf-8-sig")
df4 = pd.read_csv("정제결과_멘티.csv", encoding="utf-8-sig")
print(df4.shape, df4.isnull().sum().sum(), df4.duplicated().sum().sum())
print(list(df4.columns))
