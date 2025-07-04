import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import joblib

# === 路径配置 ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
MODEL_DIR = os.path.join(CURRENT_DIR, "trained_model")
os.makedirs(MODEL_DIR, exist_ok=True)

# === 读取数据 ===
csv_path = os.path.join(ROOT_DIR, "data", "train_data", "particles_output(no_wind).csv")
print("📂 正在读取 CSV 文件...")
if not os.path.exists(csv_path):
    print(f"❌ 文件不存在: {csv_path}")
    exit()

df = pd.read_csv(csv_path)
print("✅ CSV 文件读取成功，数据行数：", len(df))
print(df.head())

# === 特征选择与清洗 ===
features = ["lon", "lat", "x_sea_water_velocity", "y_sea_water_velocity"]
target = ["lon", "lat"]
seq_len = 10

df = df.replace([np.inf, -np.inf], np.nan).dropna()
for col in features + target:
    if df[col].nunique() <= 1:
        print(f"⚠️ 列 {col} 为常数列，已删除")
        df.drop(columns=[col], inplace=True)
if df.isnull().any().any():
    raise ValueError("❌ 数据中存在 NaN，请检查！")

# === 标准化 ===
print("🔄 正在标准化数据...")
feature_scaler = StandardScaler()
target_scaler = StandardScaler()
df[features] = feature_scaler.fit_transform(df[features])
df[target] = target_scaler.fit_transform(df[target])

# === 构造序列样本 ===
print("🧱 正在构造序列样本...")
X, Y = [], []
for i in range(len(df) - seq_len):
    X.append(df[features].iloc[i:i+seq_len].values)
    Y.append(df[target].iloc[i+seq_len].values)
X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.float32)
if np.isnan(X).any() or np.isnan(Y).any():
    raise ValueError("❌ 构造后的数据中存在 NaN！")
print("✅ 构造完成，X shape:", X.shape, "Y shape:", Y.shape)

# === 数据划分与加载 ===
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
class DriftDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X)
        self.Y = torch.tensor(Y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

train_loader = DataLoader(DriftDataset(X_train, Y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(DriftDataset(X_val, Y_val), batch_size=32)
print("✅ 数据加载完成，训练样本数：", len(train_loader.dataset), "验证样本数：", len(val_loader.dataset))

# === Transformer 模型结构 ===
class DriftTransformer(nn.Module):
    def __init__(self, input_dim=4, model_dim=128, nhead=8, num_layers=4, output_dim=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(model_dim, output_dim)
    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return torch.tanh(self.output(x))

# === 初始化模型与训练参数 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DriftTransformer().to(device)
print("✅ 模型初始化完成，设备：", device)
print(model)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# === 模型训练 ===
print("🚀 开始训练...")
for epoch in range(1, 11):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        if torch.isnan(loss):
            print("❌ NaN Loss 检测到！")
            exit()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)

    # === 验证阶段 ===
    model.eval()
    val_loss = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            val_loss += criterion(pred, yb).item()
            all_preds.append(pred.cpu().numpy())
            all_targets.append(yb.cpu().numpy())
    avg_val_loss = val_loss / len(val_loader)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # === 反归一化 & 评估精度 ===
    all_preds_inv = target_scaler.inverse_transform(all_preds)
    all_targets_inv = target_scaler.inverse_transform(all_targets)
    mae = mean_absolute_error(all_targets_inv, all_preds_inv)
    rmse = np.sqrt(mean_squared_error(all_targets_inv, all_preds_inv))

    print(f"✅ Epoch {epoch:2d}: Train Loss = {avg_loss:.6f}, Val Loss = {avg_val_loss:.6f} | MAE = {mae:.6f}, RMSE = {rmse:.6f}")

    # === 保存模型与 scaler ===
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "drift_transformer.pth"))
    joblib.dump(feature_scaler, os.path.join(MODEL_DIR, "feature_scaler.save"))
    joblib.dump(target_scaler, os.path.join(MODEL_DIR, "target_scaler.save"))
