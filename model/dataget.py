#!/usr/bin/env python3
# example_Japan.py

from datetime import timedelta
import numpy as np
import os
import pandas as pd
from opendrift.readers import reader_netCDF_CF_generic
from opendrift.models.oceandrift import OceanDrift

# 初始化模型
o = OceanDrift(loglevel=20)

# 设置 NetCDF 数据路径
reader_current = reader_netCDF_CF_generic.Reader(
    r"F:\open_drifter\opendrift\tests\test_data\21Japan\Japan_21-22.nc"
)
reader_wind = reader_netCDF_CF_generic.Reader(
    r"F:/open_drifter/opendrift/tests/test_data/20Japan/era5_wind_ready.nc"
)

# 加载 readers
o.add_reader([reader_current, reader_wind])

# 模型配置
o.set_config("drift:vertical_mixing", False)

# 投放粒子 - 近日本本州新潟沿海（可见陆地）
n_elements = 2000
wind_drift_factor = np.random.uniform(0, 0.06, n_elements)
o.seed_elements(
    lon=141.0, lat=37.5,          # 新潟附近海域
    radius=50000,                # 半径20km 
    number=n_elements,
    time=reader_current.start_time,
    wind_drift_factor=wind_drift_factor
)

# 模拟运行72小时
o.run(
    time_step=timedelta(minutes=15),
    time_step_output=timedelta(hours=1),
    duration=timedelta(hours=720)
)

# 输出模拟信息
print(o)
time_array = o.get_time_array()
print(f"Start time: {time_array[0]}")
print(f"End time: {time_array[-1]}")
print(f"Output steps: {len(time_array)}")

# 取消动图生成，转为向量csv模型
lons = o.result.lon.values
lats = o.result.lat.values
x_curr = o.result.x_sea_water_velocity.values
y_curr = o.result.y_sea_water_velocity.values
times = o.result.time.values  # 这里是1维时间数组

n_particles, n_times = lons.shape

records = []
for pid in range(n_particles):
    for tid in range(n_times):
        records.append({
            'trajectory': pid,
            'time': str(times[tid]),  # 建议转成字符串
            'lon': lons[pid, tid],
            'lat': lats[pid, tid],
            'x_sea_water_velocity': x_curr[pid, tid],
            'y_sea_water_velocity': y_curr[pid, tid],
        })

df = pd.DataFrame(records)
df.to_csv(r'F:\open_drifter\result\lizi_result\Japan6to7.csv', index=False)
print('导出完成:', df.shape)