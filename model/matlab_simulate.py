import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
import os

# 设置全局DPI为650
plt.rcParams['figure.dpi'] = 650
plt.rcParams['savefig.dpi'] = 650

csv_path = r'F:\open_drifter\transform_drift\result\Japan6to7.csv'
df = pd.read_csv(csv_path, parse_dates=["time"])

# 数据预处理
df = df[['trajectory', 'time', 'lon', 'lat']].sort_values(by=["trajectory", "time"])
unique_times = sorted(df['time'].unique())

# 输出目录设置
output_dir = r"F:\open_drifter\result\figure2"
os.makedirs(output_dir, exist_ok=True)

# 1. 高DPI初始投放图
fig_init = plt.figure(figsize=(7, 7), dpi=650)
ax_init = plt.axes(projection=ccrs.PlateCarree())
ax_init.set_extent([137, 160, 33, 44], crs=ccrs.PlateCarree())
ax_init.add_feature(cfeature.LAND.with_scale('50m'), facecolor='lightgray')
ax_init.coastlines(resolution='50m')

# 网格线配置
gl_init = ax_init.gridlines(draw_labels=True)
gl_init.top_labels = False
gl_init.right_labels = False
gl_init.left_labels = True
gl_init.bottom_labels = True

initial_df = df[df['time'] == unique_times[0]]
ax_init.scatter(
    initial_df['lon'], 
    initial_df['lat'], 
    color='#CCBBFF', 
    s=15,  # 增大点大小以适应高DPI
    edgecolor='none',
    alpha=0.7
)
ax_init.set_title(f"Initial Particle Distribution\n{unique_times[0]}", pad=20)

# 保存初始图
initial_plot_path = os.path.join(output_dir, "Japan6to7(matlab).png")
fig_init.savefig(
    initial_plot_path, 
    dpi=650,
    bbox_inches='tight',
    # 移除 quality 参数（仅JPEG需要）
    metadata={'CreationDate': None}
)
plt.close(fig_init)
print(f"✅ 初始投放图(650DPI)已保存到: {initial_plot_path}")

# 2. 高DPI动画设置
fig_ani = plt.figure(figsize=(7, 7), dpi=650)
ax_ani = plt.axes(projection=ccrs.PlateCarree())
ax_ani.set_extent([137, 160, 33, 44], crs=ccrs.PlateCarree())
ax_ani.add_feature(cfeature.LAND.with_scale('50m'), facecolor='lightgray')
ax_ani.coastlines(resolution='50m')

# 动画元素
scat = ax_ani.scatter(
    [], [], 
    color='#99FFFF', 
    s=15,
    edgecolor='none',
    alpha=0.7
)
timestamp = ax_ani.text(
    0.5, 1.02, '',
    transform=ax_ani.transAxes,
    ha='center',
    fontsize=12
)

def update(frame):
    current_time = unique_times[frame]
    sub_df = df[df['time'] == current_time]
    scat.set_offsets(sub_df[['lon', 'lat']].values)
    timestamp.set_text(current_time.strftime("%Y-%m-%d %H:%M"))
    return scat, timestamp

# 创建动画
ani = FuncAnimation(
    fig_ani,
    update,
    frames=len(unique_times),
    interval=200,
    blit=False
)

# 保存高DPI动画
animation_path = os.path.join(output_dir, "Japan6to7(matlab).mp4")  # 改为MP4格式支持高DPI
ani.save(
    animation_path,
    writer='ffmpeg',
    fps=10,
    dpi=650,
    bitrate=5000,
    extra_args=['-preset', 'slow', '-crf', '18']  # 高质量编码
)
plt.close(fig_ani)
print(f"✅ 动画(650DPI)已保存到: {animation_path}")