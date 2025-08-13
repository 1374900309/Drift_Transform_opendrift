#!/usr/bin/env python3
# drift_fukushima_yearlong.py

from datetime import timedelta
import numpy as np
import os
from opendrift.readers import reader_netCDF_CF_generic
from opendrift.models.oceandrift import OceanDrift

# 初始化模型
o = OceanDrift(loglevel=20)

# NetCDF 数据路径（确保这些文件存在）
reader_current = reader_netCDF_CF_generic.Reader(
    r"F:\open_drifter\opendrift\tests\test_data\French\French_3_to_4.nc"
)
reader_wind = reader_netCDF_CF_generic.Reader(
    r"F:/open_drifter/opendrift/tests/test_data/20Japan/era5_wind_ready.nc"
)

# 添加 readers
o.add_reader([reader_current, reader_wind])

# 配置模型
o.set_config("drift:vertical_mixing", False)

# 粒子设置 - 福岛外海（经纬度大致为 141°E, 37.5°N）
n_elements = 2000
wind_drift_factor = np.random.uniform(0, 0.06, n_elements)

o.seed_elements(
    lon=-138.8, lat=-21.9,          
    radius=2000,                 
    number=n_elements,
    time=reader_current.start_time,
    wind_drift_factor=wind_drift_factor
)

# 模拟一年
o.run(
    time_step=timedelta(minutes=15),
    time_step_output=timedelta(hours=1),
    duration=timedelta(days=7)
)

# 打印输出信息
print(o)
time_array = o.get_time_array()
print(f"Start time: {time_array[0]}")
print(f"End time: {time_array[-1]}")
print(f"Output steps: {len(time_array)}")

# 输出路径设置
output_dir = r"F:\open_drifter\result\figure"
os.makedirs(output_dir, exist_ok=True)
gif_path = os.path.join(output_dir, "French.gif")
png_path = os.path.join(output_dir, "French.png")

o.animation(
    color="wind_drift_factor",
    fast=False,
    show_time=True,
    show_landmask=True,
    landmask_resolution="50m",
    filename=gif_path,
    view=[-140.5, -137.5, -23, -20],        
    auto_range=False
)

# 生成静图
o.plot(
    linecolor="wind_drift_factor",
    fast=True,
    filename=png_path,
    show_landmask=True,
    landmask_resolution="50m",
    lonlat_box=[-140.5, -137.5, -23, -20],
    auto_range=False
)


print("✅ 动图保存路径:", gif_path)
print("✅ 静图保存路径:", png_path)
