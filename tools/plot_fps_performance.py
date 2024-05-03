import matplotlib.pyplot as plt
# 设置全局颜色
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['steelblue'])

fps = [2.1, 5.4, 3.2, 7.6]
ray_iou = [32.4, 29.6, 32.6, 33.5]
labels = ['BEVFormer', 'BEVDet-Occ', 'BEVDet-Occ-\nLongterm', 'FB-Occ']

# 绘制散点图
plt.scatter(fps, ray_iou)

# 添加文本
for i in range(len(fps)):
    if labels[i] == 'BEVDet-Occ-\nLongterm':
        plt.text(fps[i], ray_iou[i]+0.4, labels[i], fontsize=10, ha='center', va='bottom')  # 通过减去0.5调整文本位置
    elif labels[i] == 'BEVFormer':
        plt.text(fps[i]+0.5, ray_iou[i]-0.4, labels[i], fontsize=10, ha='center', va='top')  # 通过减去0.5调整文本位置
    else:
        plt.text(fps[i], ray_iou[i]-0.4, labels[i], fontsize=10, ha='center', va='top')  # 通过减去0.5调整文本位置


# SparseOCC
fps = [17.3, 12.5]
ray_iou = [34.0, 35.1]
labels = ['SparseOcc(8f)', 'SparseOcc(16f)']
plt.scatter(fps, ray_iou, color=None)
# 添加文本
plt.text(fps[0], ray_iou[0]-0.2, labels[0], fontsize=10, ha='center', va='top')
plt.text(fps[1], ray_iou[1]+0.2, labels[1], fontsize=10, ha='center', va='bottom')
# 连接散点并画线
plt.plot(fps, ray_iou, color=None, linestyle='-')  # 修改线型

# FlashOccV2
# fps = [29.0, 22.6, 22.0, 20.3]
# ray_iou = [34.57, 34.93, 35.99, 38.51]
# labels = ['FlashOccV2-\nTiny(1f)', 'FlashOccV2(1f)', 'FlashOccV2(2f)', 'FlashOccV2(8f)']
fps = [29.0, 22.6, 22.0, 20.3]
ray_iou = [34.81, 35.22, 36.76, 39.12]
labels = ['FlashOccV2-\nTiny(1f)', 'FlashOccV2(1f)', 'FlashOccV2(2f)', 'FlashOccV2(8f)']
plt.scatter(fps, ray_iou, color='orange')
# 添加文本
plt.text(fps[0], ray_iou[0]-0.4, labels[0], fontsize=10, ha='center', va='top')
plt.text(fps[1], ray_iou[1]-0.4, labels[1], fontsize=10, ha='center', va='top')
plt.text(fps[2]+0.4, ray_iou[2], labels[2], fontsize=10, ha='left', va='bottom')
plt.text(fps[3], ray_iou[3]+0.4, labels[3], fontsize=10, ha='center', va='bottom')
# 连接散点并画线
plt.plot(fps, ray_iou, color='orange', linestyle='-')  # 修改线型


# 设置字体大小和粗细
font = {'family': 'times new roman',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }
# 设置图表标题和坐标轴标签
plt.xlabel('FPS (Frames Per Second)', fontdict=font)
plt.ylabel('RayIoU', fontdict=font)

# 设置 y 轴范围
plt.ylim(28, 40)
# 设置 y 轴刻度
plt.yticks([30, 32, 34, 36, 38, 40])

# 设置 x 轴范围
plt.xlim(0, 32)
# 设置 y 轴刻度
plt.xticks([0, 5, 10, 15, 20, 25, 30])

# 保存图像
plt.savefig('scatter_plot.png')
# 显示图表
plt.show()
