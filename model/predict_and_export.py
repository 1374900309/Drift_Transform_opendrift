import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from transformer_model import DriftTransformer
import joblib
import os

# === 参数 ===
model_path = "drift_transformer.pth"
feature_scaler_path = "feature_scaler.save"
target_scaler_path = "target_scaler.save"
csv_input_path = "predict_input.csv"  # 输入所有粒子最近10小时轨迹数据
output_csv_path = r"F:\open_drifter\transform_drift\result\predict_output_batch.csv"

# === 加载模型 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DriftTransformer(input_dim=4).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("✅ 模型加载成功")

# === 加载标准化器 ===
feature_scaler = joblib.load(feature_scaler_path)
target_scaler = joblib.load(target_scaler_path)
print("✅ 标准化器加载成功")

# === 读取数据 ===
if not os.path.exists(csv_input_path):
    raise FileNotFoundError(f"❌ 输入文件不存在: {csv_input_path}")
df = pd.read_csv(csv_input_path)
features = ["lon", "lat", "x_sea_water_velocity", "y_sea_water_velocity"]

# === 分组每个粒子的序列 ===
particle_sequences = []
num_particles = df.shape[0] // 10  # 假设每10行是一个粒子
if df.shape[0] % 10 != 0:
    raise ValueError("❌ 输入行数不是10的倍数，请确认每个粒子有10步")

for i in range(num_particles):
    segment = df.iloc[i*10:(i+1)*10][features].values
    segment_scaled = feature_scaler.transform(segment).astype(np.float32)
    particle_sequences.append(segment_scaled)

X_input_batch = np.stack(particle_sequences)  # shape: [100, 10, 4]

# === 模型预测 ===
with torch.no_grad():
    input_tensor = torch.tensor(X_input_batch).to(device)
    pred_scaled = model(input_tensor).cpu().numpy()  # shape: [100, 2]

# === 反标准化 ===
pred_real = target_scaler.inverse_transform(pred_scaled)

# === 保存结果 ===
df_out = pd.DataFrame(pred_real, columns=["pred_lon", "pred_lat"])
df_out.to_csv(output_csv_path, index=False)
print(f"✅ 已保存 {num_particles} 个粒子的预测结果到：{output_csv_path}")
