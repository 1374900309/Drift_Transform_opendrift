import pandas as pd
from datetime import datetime, timedelta
from opendrift.models.oceandrift import OceanDrift
from opendrift.readers import reader_netCDF_CF_generic
import os

# 初始化模型
o = OceanDrift(loglevel=20)

# 加载流场和风场
reader_current = reader_netCDF_CF_generic.Reader(
    r"F:\open_drifter\opendrift\tests\test_data\20Japan\Ocean_Japan_2023-8-24to2024-9-24.nc"
)
reader_wind = reader_netCDF_CF_generic.Reader(
    r"F:/open_drifter/opendrift/tests/test_data/20Japan/era5_wind_ready.nc"
)
o.add_reader([reader_current, reader_wind])

# 读取预测轨迹 CSV
csv_path = r"F:\open_drifter\result\lizi_result\particles_output(no_wind)_clean.csv"
print("📂 读取预测轨迹 CSV...")
df = pd.read_csv(csv_path, parse_dates=["time"])

# 检查列是否完整
required_columns = ["lon", "lat", "time"]
if not all(col in df.columns for col in required_columns):
    raise ValueError(f"❌ CSV 缺少列: {required_columns}")

# 设置模拟参数
o.set_config("drift:vertical_mixing", False)

# 使用所有粒子中最早的时间作为统一投放时间（兼容旧版本）
start_time = df["time"].min()

o.seed_elements(
    lon=df["lon"].values,
    lat=df["lat"].values,
    time=df["time"].values
)

tart_time = df["time"].min()
end_time = df["time"].max()
total_duration = end_time - start_time

# 设置模型起始时间（兼容 OpenDrift 1.14.x）
o.start_time = pd.to_datetime(start_time)

# 运行模拟
o.run(
    duration=total_duration + timedelta(days=2),
    time_step=timedelta(minutes=15),
    time_step_output=timedelta(hours=1)
)
# 输出动图和静图
output_dir = r"F:\open_drifter\result\figure"
os.makedirs(output_dir, exist_ok=True)

gif_path = os.path.join(output_dir, "predicted_particles_yuce1.gif")
png_path = os.path.join(output_dir, "predicted_particles_yuce1.png")

o.animation(
    filename=gif_path,
    show_time=True,
    show_landmask=True,
    landmask_resolution="50m",
    view=[130, 145, 30, 45],
    fast=True
)

o.plot(
    filename=png_path,
    show_landmask=True,
    landmask_resolution="50m",
    lonlat_box=[130, 145, 30, 45],
    fast=True
)

print("✅ 动图保存路径:", gif_path)
print("✅ 静图保存路径:", png_path)
print("✅ 模拟中粒子总数:", o.num_elements_total())
print("✅ 投放粒子数量:", len(df))
