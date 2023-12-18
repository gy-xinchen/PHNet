import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

# Set the font family to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'


# 创建数据
data1 = np.array([[0.847, 0.435],
                  [0.359, 0.830]])

data2 = np.array([[0.855, 0.542],
                  [0.504, 0.794]])

data3 = np.array([[0.964, 0.732],
                  [0.828, 0.901]])



# 找到两个表格的最小值和最大值
min_value = min(data1.min(), data2.min(), data3.min())
max_value = max(data1.max(), data2.max(), data3.max())


# 定义颜色映射，从蓝到红的渐变
colors = [(63/255, 127/255, 147/255), (1, 1, 1), (193/255, 81/255, 52/255)]
cmap_name = 'custom_colormap'
cm = LinearSegmentedColormap.from_list(cmap_name, colors)

# 创建图表
fig = plt.figure(figsize=(12, 5))
gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.1)  # 减小水平间距



# 绘制第一个表格
ax1 = plt.subplot(gs[0])
cax1 = ax1.matshow(data1, cmap=cm, vmin=min_value, vmax=max_value)
# 定义行和列的标签
row_labels = ['Ours', 'ShefPAH-179']
column_labels = ['Ours', 'ShefPAH-179']
# 对位置进行循环以获取标题
for i in range(len(row_labels)):
    for j in range(len(column_labels)):
        # 获取当前位置的标题
        title = 'ML MPCA-lr {} - {}'.format(row_labels[i], column_labels[j])
        # 设置当前位置的标题
        ax1.set_title(title, pad=10)  # 使用pad参数来控制标题与图表的距离

        # 设置x和y轴标签
        ax1.set_xticks(range(len(column_labels)))
        ax1.set_xticklabels(column_labels, fontsize=12)
        ax1.set_yticks(range(len(row_labels)))
        ax1.set_yticklabels(row_labels, fontsize=12)
ax1.set_title('ML MPCA-LR')
ax1.set_ylabel('Training vendor', fontsize=16)
ax1.set_xlabel('Test vendor', fontsize=16)

# 在每个条形的顶部添加数值标签，根据数据值大小设置标签颜色
for i in range(len(data1)):
    for j in range(len(data1[i])):
        label_color = 'black' if data1[i][j] > 0.6 else 'white'
        ax1.text(j, i, f'{data1[i][j]:.3f}', ha='center', va='center', color=label_color, fontsize=16)



# 绘制第二个表格
ax2 = plt.subplot(gs[1])
cax2 = ax2.matshow(data2, cmap=cm, vmin=min_value, vmax=max_value)
# 定义行和列的标签
row_labels2 = [' ', ' ']
column_labels2 = [' ', ' ']
# 对位置进行循环以获取标题
for i in range(len(row_labels2)):
    for j in range(len(column_labels2)):
        # 获取当前位置的标题
        title = 'ML Texture-MLP {} - {}'.format(row_labels2[i], column_labels2[j])
        # 设置当前位置的标题
        ax2.set_title(title, pad=10)  # 使用pad参数来控制标题与图表的距离

        # 设置x和y轴标签
        ax2.set_xticks(range(len(column_labels2)))
        ax2.set_xticklabels(column_labels2)
        ax2.set_yticks(range(len(row_labels2)))
        ax2.set_yticklabels(row_labels2)
ax2.set_title('ML Texture-MLP')

# 在每个条形的顶部添加数值标签，根据数据值大小设置标签颜色
for i in range(len(data2)):
    for j in range(len(data2[i])):
        label_color = 'black' if data2[i][j] > 0.6 else 'white'
        ax2.text(j, i, f'{data2[i][j]:.3f}', ha='center', va='center', color=label_color, fontsize=16)

# 绘制第三个表格
ax3 = plt.subplot(gs[2])
cax3 = ax3.matshow(data3, cmap=cm, vmin=min_value, vmax=max_value)
# 定义行和列的标签
row_labels3 = [' ', ' ']
column_labels3 = [' ', ' ']
# 对位置进行循环以获取标题
for i in range(len(row_labels3)):
    for j in range(len(column_labels3)):
        # 获取当前位置的标题
        title = 'ML Texture-MLP {} - {}'.format(row_labels3[i], column_labels3[j])
        # 设置当前位置的标题
        ax3.set_title(title, pad=10)  # 使用pad参数来控制标题与图表的距离

        # 设置x和y轴标签
        ax3.set_xticks(range(len(column_labels3)))
        ax3.set_xticklabels(column_labels3)
        ax3.set_yticks(range(len(row_labels3)))
        ax3.set_yticklabels(row_labels3)
ax3.set_title('DL PAHNet')

# 在每个条形的顶部添加数值标签，根据数据值大小设置标签颜色
for i in range(len(data3)):
    for j in range(len(data3[i])):
        label_color = 'black' if data3[i][j] > 0.6 else 'white'
        ax3.text(j, i, f'{data3[i][j]:.3f}', ha='center', va='center', color=label_color, fontsize=16)

# 添加共享颜色条
cax4 = plt.subplot(gs[3])
cbar = plt.colorbar(cax3, cax=cax4, orientation='vertical')
ticks = np.arange(min_value, max_value, 0.1)  # 生成刻度值
cbar.set_ticks(ticks)
cbar.set_ticklabels([f'{tick:.1f}' for tick in ticks])  # 将刻度值转为字符串并设置为标签



# 调整布局
plt.tight_layout()

# 显示图表
plt.savefig(r"G:\CMR-res\muti_center_data0927\Data_reword\ismsm會議統計\transfer1.svg")
