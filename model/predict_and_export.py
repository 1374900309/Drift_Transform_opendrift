import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from transformer_model import DriftTransformer
import joblib
import os


# 参数导入
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
TRAINED_MODEL_DIR = os.path.join(CURRENT_DIR, "trained_model")

model_path = os.path.join(TRAINED_MODEL_DIR, "drift_transformer.pth")
feature_scaler_path = os.path.join(TRAINED_MODEL_DIR, "feature_scaler.save")
target_scaler_path = os.path.join(TRAINED_MODEL_DIR, "target_scaler.save")
csv_input_path = os.path.join(ROOT_DIR, "data", "for_predict", "predict_input.csv")  # 用于生成预测数据的数据 10；1预测粒子位移
output_csv_path = os.path.join(ROOT_DIR, "result", "prediction_data", "predict_output_batch.csv") # 生成预测后的粒子数据

# 加载模型 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DriftTransformer(input_dim=4).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("✅ 模型加载成功")

# 加载标准化器 
feature_scaler = joblib.load(feature_scaler_path)
target_scaler = joblib.load(target_scaler_path)
print("✅ 标准化器加载成功")

# 读取数据 
if not os.path.exists(csv_input_path):
    raise FileNotFoundError(f"❌ 输入文件不存在: {csv_input_path}")
df = pd.read_csv(csv_input_path)
features = ["lon", "lat", "x_sea_water_velocity", "y_sea_water_velocity"]

# 分组每个粒子的序列 
particle_sequences = []
num_particles = df.shape[0] // 10  
if df.shape[0] % 10 != 0:
    raise ValueError("❌ 输入行数不是10的倍数，请确认每个粒子有10步")

for i in range(num_particles):
    segment = df.iloc[i*10:(i+1)*10][features].values
    segment_scaled = feature_scaler.transform(segment).astype(np.float32)
    particle_sequences.append(segment_scaled)

X_input_batch = np.stack(particle_sequences)  

# 模型预测
with torch.no_grad():
    input_tensor = torch.tensor(X_input_batch).to(device)
    pred_scaled = model(input_tensor).cpu().numpy()  

# 反标准化 
pred_real = target_scaler.inverse_transform(pred_scaled)

# 保存结果 
RESULT_DIR = os.path.join(CURRENT_DIR, "result", "prediction_data")
os.makedirs(RESULT_DIR, exist_ok=True)
df_out = pd.DataFrame(pred_real, columns=["pred_lon", "pred_lat"])
df_out.to_csv(output_csv_path, index=False)
print(f"✅ 已保存 {num_particles} 个粒子的预测结果到：{output_csv_path}")
