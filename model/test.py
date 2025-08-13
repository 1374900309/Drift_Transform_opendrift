import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

# 文件路径和保存路径
file_path = r'F:\open_drifter\opendrift\tests\test_data\20Japan\Ocean_Japan_2023-8-24to2024-9-24.nc'
save_path = r'F:\open_drifter\result\figure\OceanCurrent_UComponent.png'

# 打开 NetCDF 数据集
ds = xr.open_dataset(file_path)
print(ds.data_vars)

# ✅ 取出 U 分量的表层 (depth=0) 和第一个时间点
u = ds['uo'].isel(time=0, depth=0)  # 变成 (lat, lon)
lon = ds['longitude']
lat = ds['latitude']

# 绘图
fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
cf = ax.contourf(lon, lat, u, 60, transform=ccrs.PlateCarree(), cmap='RdBu_r')
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.set_title('Surface Ocean Current (U Component)', fontsize=14)

# 颜色条
cbar = plt.colorbar(cf, orientation='vertical', pad=0.03)
cbar.set_label('U Current Velocity [m/s]', fontsize=12)

# 保存图像
plt.tight_layout()
plt.savefig(save_path, dpi=300)
print(f'✅ 图像已保存至: {save_path}')
plt.close()
