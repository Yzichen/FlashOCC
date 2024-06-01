import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist
# fig, _ = plt.subplots(3, 1, figsize=(5, 12))
fig, _ = plt.subplots(1, 3, figsize=(15, 5))
fig.set_tight_layout(True)
# 设置全局颜色
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['steelblue'])

# plt.subplot(3, 1, 3)
plt.subplot(1, 3, 3)


# ax = axisartist.Subplot(fig, 111)
# #将绘图区对象添加到画布中
# fig.add_axes(ax)
# #通过set_axisline_style方法设置绘图区的底部及左侧坐标轴样式
# #"-|>"代表实心箭头："->"代表空心箭头
# ax.axis["bottom"].set_axisline_style("->", size = 1.5)
# ax.axis["left"].set_axisline_style("->", size = 1.5)
# #通过set_visible方法设置绘图区的顶部及右侧坐标轴隐藏
# ax.axis["top"].set_visible(False)
# ax.axis["right"].set_visible(False)
                        

# SparseOCC
fps = [17.3]
ray_iou = [14.1]
labels = ['SparseOcc(8f)']
plt.scatter(fps, ray_iou, color='dodgerblue')
# 添加文本
plt.text(fps[0]+1.5, ray_iou[0]-0.1, labels[0], fontsize=10, ha='center', va='top')

# Panoptic-FlashOcc
# fps = [29.0, 22.6, 22.0, 20.3] # 3090
fps = [27.3, 26.1, 25.7, 24.8] # a100 40g
# ray_iou = [12.6, 12.9, 14.2, 15.8]
ray_iou = [12.89, 13.18, 14.52, 15.96]
labels = ['Panoptic-\nFlashOcc-Tiny(1f)', 'Panoptic-\nFlashOcc(1f)', 'Panoptic-\nFlashOcc(2f)', 'Panoptic-\nFlashOcc(8f)']
plt.scatter(fps, ray_iou, color='orange')
# 添加文本
plt.text(fps[0]-.5, ray_iou[0]-0.2, labels[0], fontsize=10, ha='center', va='top')
plt.text(fps[1]+2.0, ray_iou[1]+0.3, labels[1], fontsize=10, ha='center', va='top')
plt.text(fps[2]+0.4, ray_iou[2], labels[2], fontsize=10, ha='left', va='bottom')
plt.text(fps[3]+2., ray_iou[3]-0.1, labels[3], fontsize=10, ha='center', va='bottom')
# 连接散点并画线
plt.plot(fps, ray_iou, color='orange', linestyle='-')  # 修改线型
plt.grid(True)
plt.grid(color='gray', linestyle='--', linewidth=1, alpha=0.3)

# 设置字体大小和粗细
font = {'family': 'times new roman',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }
# 设置图表标题和坐标轴标签
plt.xlabel('FPS', fontdict=font)
plt.ylabel('Occ3D-nuScenes (RayPQ)', fontdict=font)

# 设置 y 轴范围
plt.ylim(11.5, 16.5)
# 设置 y 轴刻度
plt.yticks([12, 13, 14, 15, 16])

# 设置 x 轴范围
plt.xlim(16, 30)
# 设置 y 轴刻度
plt.xticks([16, 20, 24, 28])


# plt.subplot(3, 1, 2)
plt.subplot(1, 3, 2)
# BEVFormer
fps = [3.0]
ray_iou = [23.7]
labels = ['BEVFormer']
plt.scatter(fps, ray_iou, color='dodgerblue')
# 添加文本
plt.text(fps[0]+3.2, ray_iou[0]+0.2, labels[0], fontsize=10, ha='center', va='top')

# FB-Occ
fps = [10.3]
ray_iou = [27.9]
labels = ['FB-Occ']
plt.scatter(fps, ray_iou, color='dodgerblue')
# 添加文本
plt.text(fps[0], ray_iou[0]-0.2, labels[0], fontsize=10, ha='center', va='top')

# SparseOCC
fps = [17.3, 12.5]
ray_iou = [30.3, 30.9]
labels = ['SparseOcc(8f)', 'SparseOcc(16f)']
plt.scatter(fps, ray_iou, color='dodgerblue')
# 添加文本
plt.text(fps[0], ray_iou[0]-0.2, labels[0], fontsize=10, ha='center', va='top')
plt.text(fps[1], ray_iou[1]+0.2, labels[1], fontsize=10, ha='center', va='bottom')
# 连接散点并画线
plt.plot(fps, ray_iou, color='dodgerblue', linestyle='-')  # 修改线型

# Panoptic-FlashOcc
# fps = [29.0, 22.6, 22.0, 20.3] # 3090
fps = [33.4, 30.6, 30.1, 29.1] # a100 40g
ray_iou = [29.1, 29.4, 30.3, 31.6]
labels = ['Panoptic-\nFlashOcc-Tiny(1f)', 'Panoptic-\nFlashOcc(1f)', 'Panoptic-\nFlashOcc(2f)', 'Panoptic-\nFlashOcc(8f)']
plt.scatter(fps, ray_iou, color='orange')
# 添加文本
plt.text(fps[0]-2.0, ray_iou[0]-0.2, labels[0], fontsize=10, ha='center', va='top')
plt.text(fps[1]-3.5, ray_iou[1]+0.4, labels[1], fontsize=10, ha='center', va='top')
plt.text(fps[2]-7.0, ray_iou[2]-0.3, labels[2], fontsize=10, ha='left', va='bottom')
plt.text(fps[3]-4.0, ray_iou[3]-0.5, labels[3], fontsize=10, ha='center', va='bottom')
# 连接散点并画线
plt.plot(fps, ray_iou, color='orange', linestyle='-')  # 修改线型
plt.grid(True)
plt.grid(color='gray', linestyle='--', linewidth=1, alpha=0.3)

# 设置字体大小和粗细
font = {'family': 'times new roman',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }
# 设置图表标题和坐标轴标签
plt.xlabel('FPS', fontdict=font)
plt.ylabel('Occ3D-nuScenes (mIoU)', fontdict=font)

# 设置 y 轴范围
plt.ylim(23, 33)
# 设置 y 轴刻度
plt.yticks([24, 26, 28, 30, 32])

# 设置 x 轴范围
plt.xlim(2, 35)
# 设置 y 轴刻度
plt.xticks([5, 10, 15, 20, 25, 30, 35])




# plt.subplot(3, 1, 1)
plt.subplot(1, 3, 1)
fps = [2.1, 5.4, 3.2, 7.6]
ray_iou = [32.4, 29.6, 32.6, 33.5]
labels = ['BEVFormer', 'BEVDet-Occ', 'BEVDet-Occ-\nLongterm', 'FB-Occ']

# 绘制散点图
plt.scatter(fps, ray_iou, color='dodgerblue')

# 添加文本
for i in range(len(fps)):
    if labels[i] == 'BEVDet-Occ-\nLongterm':
        plt.text(fps[i]+0.3, ray_iou[i]+0.2, labels[i], fontsize=10, ha='center', va='bottom')  # 通过减去0.5调整文本位置
    elif labels[i] == 'BEVFormer':
        plt.text(fps[i]+1.0, ray_iou[i]-0.2, labels[i], fontsize=10, ha='center', va='top')  # 通过减去0.5调整文本位置
    elif labels[i] == 'BEVDet-Occ':
        plt.text(fps[i]+0.2, ray_iou[i]+0.5, labels[i], fontsize=10, ha='center', va='top')  # 通过减去0.5调整文本位置
    else:
        plt.text(fps[i]+0.2, ray_iou[i]+0.4, labels[i], fontsize=10, ha='center', va='top')  # 通过减去0.5调整文本位置


# SparseOCC
fps = [17.3, 12.5]
ray_iou = [34.0, 35.1]
labels = ['SparseOcc(8f)', 'SparseOcc(16f)']
plt.scatter(fps, ray_iou, color='dodgerblue')
# 添加文本
plt.text(fps[0], ray_iou[0]-0.2, labels[0], fontsize=10, ha='center', va='top')
plt.text(fps[1], ray_iou[1]+0.2, labels[1], fontsize=10, ha='center', va='bottom')
# 连接散点并画线
plt.plot(fps, ray_iou, color='dodgerblue', linestyle='-')  # 修改线型

# Panoptic-FlashOcc
# fps = [29.0, 22.6, 22.0, 20.3]
fps = [33.4, 30.6, 30.1, 29.1] # a100 40g
ray_iou = [34.81, 35.22, 36.76, 38.50]
labels = ['Panoptic-\nFlashOcc-Tiny(1f)', 'Panoptic-\nFlashOcc(1f)', 'Panoptic-\nFlashOcc(2f)', 'Panoptic-\nFlashOcc(8f)']
plt.scatter(fps, ray_iou, color='orange')
# 添加文本
plt.text(fps[0]-4.0, ray_iou[0]+0.0, labels[0], fontsize=10, ha='center', va='top')
plt.text(fps[1]-4.2, ray_iou[1]+0.4, labels[1], fontsize=10, ha='center', va='top')
plt.text(fps[2]-8.5, ray_iou[2]-0.3, labels[2], fontsize=10, ha='left', va='bottom')
plt.text(fps[3]-4.0, ray_iou[3]-0.5, labels[3], fontsize=10, ha='center', va='bottom')
# 连接散点并画线
plt.plot(fps, ray_iou, color='orange', linestyle='-')  # 修改线型
plt.grid(True)
plt.grid(color='gray', linestyle='--', linewidth=1, alpha=0.3)

# 设置字体大小和粗细
font = {'family': 'times new roman',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }
# 设置图表标题和坐标轴标签
plt.xlabel('FPS', fontdict=font)
plt.ylabel('Occ3D-nuScenes (RayIoU)', fontdict=font)

# 设置 y 轴范围
plt.ylim(29, 39)
# 设置 y 轴刻度
plt.yticks([30, 32, 34, 36, 38])

# 设置 x 轴范围
plt.xlim(0, 35)
# 设置 y 轴刻度
plt.xticks([0, 5, 10, 15, 20, 25, 30, 35])

# 保存图像
plt.savefig('scatter_plot.png')
plt.savefig('scatter_plot.pdf')
# 显示图表
plt.show()
