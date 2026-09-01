import numpy as np
import pandas as pd

df = pd.read_csv("로그배치1.csv", encoding="utf-8-sig")
cols = ["CPU온도", "전력", "응답시간", "메모리"]
keys = ["수집시각", "구역", "센서ID"]

print("===== 문제 1 =====")
df["전력"] = pd.to_numeric(df["전력"], errors="coerce")
df.drop_duplicates(inplace=True)
print("[keys 중복 행 개수]", df.duplicated(subset=keys).sum(), "개")
print(df[df.duplicated(subset=keys, keep=False)][keys + ["CPU온도", "메모리"]])
df.drop_duplicates(subset=keys, inplace=True)
print(
    "[표 크기]",
    df.shape,
    "/ [구역별 행 수]",
    df.groupby("구역")["수집시각"].count().to_dict(),
)
cpu_avg = df["CPU온도"].mean()
cpu_std = df["CPU온도"].std(ddof=0)
cpu_z = (df["CPU온도"] - cpu_avg) / cpu_std
cpu_gb_avg = df.groupby("구역")["CPU온도"].mean()
cpu_gb_std = df.groupby("구역")["CPU온도"].std(ddof=0)
cpu_gb_z = (df["CPU온도"] - df["구역"].map(cpu_gb_avg)) / df["구역"].map(cpu_gb_std)
print(
    "[(가)-기준]",
    len(cpu_z[abs(cpu_z) > 2.5]),
    "[(나)-기준]",
    len(cpu_gb_z[abs(cpu_gb_z) > 2.5]),
)
print(df[cpu_gb_z.abs() > 2.5][keys + ["CPU온도"]].sort_values("구역"))
print(df.groupby("구역")["CPU온도"].mean().round(2).to_dict())


print("===== 문제 2 =====")
re_cpu_gb_avg = df[cpu_gb_z.abs() <= 2.5].groupby("구역")["CPU온도"].mean()
result_df = pd.DataFrame(
    {
        "포함": cpu_gb_avg.round(3),
        "제외": re_cpu_gb_avg.round(3),
        "차이": (cpu_gb_avg - re_cpu_gb_avg).round(3),
    }
)
print(result_df)
df_remove = df.copy()
df_remove["CPU온도"] = df_remove["CPU온도"].fillna(df_remove["구역"].map(re_cpu_gb_avg))
re_mem_gb_md = df_remove.groupby("구역")["메모리"].median()
df_remove["메모리"] = df_remove["메모리"].fillna(df_remove["구역"].map(re_mem_gb_md))
re_ele_gb_md = df_remove.groupby("구역")["전력"].median()
df_remove["전력"] = df_remove["전력"].fillna(df_remove["구역"].map(re_ele_gb_md))
print("\n남은 총 결측치 개수 ", df_remove.isnull().sum().sum(), "개")
print(df_remove.groupby("구역")["CPU온도"].mean().round(2).to_dict())

mem_gb_avg = df_remove.groupby("구역")["메모리"].mean()
mem_gb_std = df_remove.groupby("구역")["메모리"].std(ddof=0)
mem_gb_z = (df_remove["메모리"] - df_remove["구역"].map(mem_gb_avg)) / df_remove[
    "구역"
].map(mem_gb_std)
print(
    "\n[Z3-찰리 메모리의 표준편차]",
    mem_gb_std.loc["Z3-찰리"].round(2),
    "\n1차 :",
    len(df_remove[mem_gb_z.abs() > 3]),
    "개",
)
print(df_remove[mem_gb_z.abs() > 3][["구역", "메모리"]])
df_remove.loc[mem_gb_z.abs() > 3, "메모리"] = df_remove["구역"].map(re_mem_gb_md)
mem_gb_avg = df_remove.groupby("구역")["메모리"].mean()
mem_gb_std = df_remove.groupby("구역")["메모리"].std(ddof=0)
mem_gb_z = (df_remove["메모리"] - df_remove["구역"].map(mem_gb_avg)) / df_remove[
    "구역"
].map(mem_gb_std)
print(
    "\n[Z3-찰리 메모리의 표준편차]",
    mem_gb_std.loc["Z3-찰리"].round(2),
    "\n2차 :",
    len(df_remove[mem_gb_z.abs() > 3]),
    "개",
)
print(df_remove[mem_gb_z.abs() > 3][["구역", "메모리"]])
df_remove.loc[mem_gb_z.abs() > 3, "메모리"] = df_remove["구역"].map(re_mem_gb_md)
mem_gb_avg = df_remove.groupby("구역")["메모리"].mean()
mem_gb_std = df_remove.groupby("구역")["메모리"].std(ddof=0)
mem_gb_z = (df_remove["메모리"] - df_remove["구역"].map(mem_gb_avg)) / df_remove[
    "구역"
].map(mem_gb_std)
print(
    "\n[Z3-찰리 메모리의 표준편차]",
    mem_gb_std.loc["Z3-찰리"].round(2),
    "\n3차 :",
    len(df_remove[mem_gb_z.abs() > 3]),
    "개",
)
print("\n", df_remove.groupby("구역")["메모리"].count().to_dict())
code_map = {"Z1-알파": 0, "Z2-브라보": 1, "Z3-찰리": 2}
code = df_remove["구역"].map(code_map)
df_remove.insert(2, "구역코드", code)  #
print(df_remove.shape)

df_remove.to_csv("정제결과_최종.csv", index=False, encoding="utf-8-sig")

print("===== 문제 3 =====")
np.random.seed(6)
mix = np.random.permutation(len(df_remove))
st_range = int(len(df_remove) * 0.6)
nd_range = int(len(df_remove) * 0.8)
train = df_remove.iloc[mix[:st_range]]
verification = df_remove.iloc[mix[st_range:nd_range]]
test = df_remove.iloc[mix[nd_range:]]
print(len(train), len(verification), len(test))

tn_mn = train[cols].min()
tn_mx = train[cols].max()
df_mn = df_remove[cols].min()
df_mx = df_remove[cols].max()
result_df2 = pd.DataFrame(
    {"학습min": tn_mn, "학습max": tn_mx, "전체min": df_mn, "전체max": df_mx}
)
print(result_df2)
result_df2.to_csv("스케일링기준.csv", encoding="utf-8-sig")
tn_scaled = (test[cols] - tn_mn) / (tn_mx - tn_mn)
df_scaled = (test[cols] - df_mn) / (df_mx - df_mn)
test_tn = (0 > tn_scaled) | (tn_scaled > 1)
test_df = (0 > df_scaled) | (df_scaled > 1)
print("전체 : ", test_df.values.sum(), "/ 학습 : ", test_tn.values.sum())
test_tn2 = (tn_scaled == 0) | (tn_scaled == 1)
test_df2 = (df_scaled == 0) | (df_scaled == 1)
print("전체 : ", test_df2.values.sum(), "/ 학습 : ", test_tn2.values.sum())
print(tn_scaled.max().round(4).to_dict())

df2 = pd.read_csv("로그배치2.csv", encoding="utf-8-sig")
dic_df2 = df2.groupby("구역")["전력"].count().to_dict()
print(dic_df2)
dic_df = df.groupby("구역")["수집시각"].count().to_dict()
lst = []
cnt = 0
for i in dic_df2:
    if i not in dic_df:
        cnt += 1
        lst.append(i)
print(cnt, lst)

df2_mn = df2.groupby("구역")["CPU온도"].min()
df2_mx = df2.groupby("구역")["CPU온도"].max()
df2_scaled = (df2["CPU온도"] - df2["구역"].map(df2_mn)) / (
    df2["구역"].map(df2_mx) - df2["구역"].map(df2_mn)
)


print("===== 문제 4 =====")
