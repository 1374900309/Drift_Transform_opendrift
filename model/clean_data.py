import pandas as pd

# 读取 CSV，强制所有内容为字符串，避免跳过伪 0
df = pd.read_csv(r"F:\open_drifter\result\lizi_result\Japan6to7.csv", dtype=str)

# 清洗伪 0：空白变 '0'，各种 0.000000 / 0 / 0.00000 变 '0'
df = df.replace(r'^\s*$', '0', regex=True)  # 空字符串 → '0'
df = df.replace(r'^0+\.?0*$', '0', regex=True)  # 如 0.00000 → '0'

# 转为数值以便比较（无法转的会变成 NaN）
df_numeric = df.apply(pd.to_numeric, errors='coerce').fillna(0)

# 从第二列开始判断是否全为 0
mask = ~((df_numeric.iloc[:, 1:] == 0).all(axis=1))

# 应用过滤掩码
df_cleaned = df[mask]

# 保存结果
df_cleaned.to_csv(r"F:\open_drifter\transform_drift\data\train_data\Japan6to7_clean.csv", index=False)
print("✅ 清洗完成，已删除所有“全为 0 或空值”的行")
