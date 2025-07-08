import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

# 文件路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
true_csv = os.path.join(ROOT_DIR, "result", "predict_target.csv")
pred_csv = os.path.join(ROOT_DIR, "result", "predict_output_batch.csv")

df_true = pd.read_csv(true_csv)
df_pred = pd.read_csv(pred_csv)

if "lon,lat" in df_true.columns:
    df_true[["lon", "lat"]] = df_true["lon,lat"].str.split("\t", expand=True).astype(float)

cols_true = ['lon', 'lat', 'x_sea_water_velocity', 'y_sea_water_velocity']
cols_pred = ['pred_lon', 'pred_lat', 'pred_x_sea_water_velocity', 'pred_y_sea_water_velocity']

print("📐 各项指标：")
for tcol, pcol in zip(cols_true, cols_pred):
    y_true = df_true[tcol].values
    y_pred = df_pred[pcol].values
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))   # <---- 这里直接用 sqrt 包裹
    print(f"  {tcol:<24}: MAE = {mae:.6f}   RMSE = {rmse:.6f}")
