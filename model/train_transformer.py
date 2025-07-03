import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import joblib

# === Step 1: 读取并清洗数据 ===
csv_path = r"F:\open_drifter\transform_drift\data\particles_output(no_wind).csv"
print("📂 正在读取 CSV 文件...")
if not os.path.exists(csv_path):
    print(f"❌ 文件不存在: {csv_path}")
    exit()

df = pd.read_csv(csv_path)
print("✅ CSV 文件读取成功，数据行数：", len(df))
print(df.head())

# 删除风场，只保留轨迹 + 流场
features = ["lon", "lat", "x_sea_water_velocity", "y_sea_water_velocity"]
target = ["lon", "lat"]
seq_len = 10

# === Step 2: 清洗数据 ===
df = df.replace([np.inf, -np.inf], np.nan).dropna()

for col in features + target:
    if df[col].nunique() <= 1:
        print(f"⚠️ 列 {col} 为常数列，已删除")
        df.drop(columns=[col], inplace=True)

# 再次检查
if df.isnull().any().any():
    raise ValueError("❌ 数据中存在 NaN，请检查！")

# === Step 3: 标准化 ===
print("🔄 正在标准化数据...")
feature_scaler = StandardScaler()
target_scaler = StandardScaler()

df[features] = feature_scaler.fit_transform(df[features])
df[target] = target_scaler.fit_transform(df[target])

# === Step 4: 构造序列数据 ===
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

# 数据划分
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# === Step 5: 构建 Dataset 和 DataLoader ===
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

# === Step 6: 模型结构 ===
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
        return torch.tanh(self.output(x))  # 限制输出范围防止爆炸

# === Step 7: 模型训练 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DriftTransformer(input_dim=4).to(device)
print("✅ 模型初始化完成，设备：", device)
print(model)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

print("🚀 开始训练...")
for epoch in range(1, 11):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)

        # 检查 NaN
        if torch.isnan(loss):
            print("❌ NaN Loss 检测到！")
            print("输入样本:", xb)
            print("模型输出:", pred)
            print("真实值:", yb)
            exit()

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)

    # 验证
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            val_loss += criterion(pred, yb).item()
    
    avg_val_loss = val_loss / len(val_loader)
    print(f"✅ Epoch {epoch:2d}: Train Loss = {avg_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
    torch.save(model.state_dict(), "drift_transformer.pth")
    joblib.dump(feature_scaler, "feature_scaler.save")
    joblib.dump(target_scaler, "target_scaler.save")
