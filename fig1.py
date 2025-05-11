import matplotlib.pyplot as plt
import numpy as np

# 1. 创建数据
x = np.linspace(-3.0, 3.0, 100)
y = np.linspace(-2.0, 2.0, 100)
X, Y = np.meshgrid(x, y)
Z = np.exp(-(X**2 + Y**2)) + np.exp(-((X - 1.5)**2 + (Y - 0.8)**2)) * 0.7

# 2. 创建图形对象
fig = plt.figure(figsize=(6, 4))
ax = fig.add_axes([0, 0, 1, 1])
ax.patch.set_alpha(0.0)  # 设置坐标轴背景透明

# 计算 Z 的范围，生成更密集的层级（如 20 层）
z_min, z_max = np.min(Z), np.max(Z)
levels = np.linspace(z_min, z_max, 20)  # 增加层级数量

# 3. 绘制等高线图，设置颜色为黑色，并指定层级
CS = ax.contour(X, Y, Z, levels=levels, colors='black')

# 隐藏坐标轴元素
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='both', which='both',
               bottom=False, top=False, left=False, right=False,
               labelbottom=False, labeltop=False, labelleft=False, labelright=False)

# 4. 保存图形
fig.savefig("dgx_full_axes.png", transparent=True, dpi=300)
print("图形已保存为 dgx_full_axes.png")