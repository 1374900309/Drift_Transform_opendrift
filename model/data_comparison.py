import pandas as pd
from sklearn.metrics import mean_absolute_error
import os

# 文件路径 
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
true_csv = os.path.join(ROOT_DIR, "result", "predict_target.csv") # 真实数据
pred_csv = os.path.join(ROOT_DIR, "result", "predict_output_batch.csv") # 模型预测数据

# 读取 CSV 
df_true = pd.read_csv(true_csv)
df_pred = pd.read_csv(pred_csv)

print("真实数据列名：", df_true.columns.tolist())
print("预测数据列名：", df_pred.columns.tolist())

# 拆分 "lon,lat" 列为两列 
if "lon,lat" in df_true.columns:
    df_true[["lon", "lat"]] = df_true["lon,lat"].str.split("\t", expand=True).astype(float)

# 计算 MAE 
lon_mae = mean_absolute_error(df_true["lon"], df_pred["pred_lon"])
lat_mae = mean_absolute_error(df_true["lat"], df_pred["pred_lat"])

print(f"📐 预测精度：")
print(f"  MAE_lon = {lon_mae:.6f}")
print(f"  MAE_lat = {lat_mae:.6f}")
