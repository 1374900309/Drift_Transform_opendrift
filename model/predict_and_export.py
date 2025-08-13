import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from transformer_model import DriftTransformer
import joblib
import os
from datetime import timedelta

# === 路径参数 ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
TRAINED_MODEL_DIR = os.path.join(CURRENT_DIR, "trained_model")

model_path = os.path.join(TRAINED_MODEL_DIR, "Japan6to7(nodandiao).pth")
feature_scaler_path = os.path.join(TRAINED_MODEL_DIR, "feature_scaler_Japan6to7(nodandiao).save")
target_scaler_path = os.path.join(TRAINED_MODEL_DIR, "target_scaler_Japan6to7(nodandiao).save")
csv_input_path = os.path.join(ROOT_DIR, "data", "for_predict", "Japan6to7_predict.csv")
output_csv_path = os.path.join(ROOT_DIR, "result", "Japan6to7_predict(nodandiao).csv")

# === 加载模型 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DriftTransformer(input_dim=4, output_dim=4).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("✅ 模型加载成功")

# === 加载标准化器 ===
feature_scaler = joblib.load(feature_scaler_path)
target_scaler = joblib.load(target_scaler_path)
print("✅ 标准化器加载成功")

# === 读取输入数据 ===
df = pd.read_csv(csv_input_path)
features = ["lon", "lat", "x_sea_water_velocity", "y_sea_water_velocity"]
seq_len = 10
pred_steps = 100

num_particles = df.shape[0] // seq_len
if df.shape[0] % seq_len != 0:
    raise ValueError("❌ 输入数据行数不是10的倍数，请确保每个粒子有10步")


base_time = pd.to_datetime("2022-06-01 00:00:00")


output_rows = []

for i in range(num_particles):
    segment = df.iloc[i * seq_len:(i + 1) * seq_len][features].values
    segment_scaled = feature_scaler.transform(segment).astype(np.float32)
    sequence = torch.tensor(segment_scaled).unsqueeze(0).to(device)

    for step in range(pred_steps):
        with torch.no_grad():
            pred_scaled = model(sequence)
        pred_real = target_scaler.inverse_transform(pred_scaled.cpu().numpy())[0]

        # 格式化时间为 ISO 纳秒精度
        timestamp = (base_time + timedelta(hours=step)).isoformat(timespec='nanoseconds')

        output_rows.append([
            i, timestamp,
            pred_real[0], pred_real[1],
            pred_real[2], pred_real[3]
        ])

        # 更新历史序列
        next_input = feature_scaler.transform(pred_real.reshape(1, -1)).astype(np.float32)
        sequence = torch.cat((sequence[:, 1:, :], torch.tensor(next_input).unsqueeze(0).to(device)), dim=1)

# === 保存为 OpenDrift 格式 CSV ===
df_out = pd.DataFrame(output_rows, columns=[
    "trajectory", "time", "lon", "lat", "x_sea_water_velocity", "y_sea_water_velocity"
])
df_out.to_csv(output_csv_path, index=False)
print(f"✅ 已保存 {num_particles} 个粒子的 100 步预测结果到：{output_csv_path}")
