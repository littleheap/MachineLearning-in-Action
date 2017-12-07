import matplotlib.pyplot as plt
from numpy import *
from kNN import kNN

# 分别获取数据矩阵和结果
print('\n获取数据……\n')
datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')

# 打印数据矩阵前20
print('\n数据矩阵前20\n')
print(datingDataMat[0:20])

# 打印类别矩阵前20
print('\n类别矩阵前20\n')
print(datingLabels[0:20])

# 制作散点图(分别以数据的第1个参数和第2个参数作为xy轴)
print('\n显示散点图……\n')
# 初始化绘制窗口
fig = plt.figure()
# 一行一列，第一个
ax = fig.add_subplot(111)
# 抽取每条数据的前两个特征作为xy轴
ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
# 显示
plt.show()

# 制作散点图(分别以数据的第2个参数和第3个参数作为xy轴)
print('\n显示散点图……\n')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
plt.show()

# 制作散点图(分别以数据的第1个参数和第3个参数作为xy轴)
print('\n显示散点图……\n')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:, 0], datingDataMat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
plt.show()

# 归一化数据处理
print('\n归一化处理数据\n')
normMat, ranges, minVals = kNN.autoNorm(datingDataMat)
print('\n归一后数据集\n')
print(normMat)
print('\n对应值域宽度大小\n')
print(ranges)
print('\n对应最小值\n')
print(minVals)

# 原始数据的90%的特征作为测试，后10%作为基础样本
kNN.datingClassTest()

# 手写数字识别系统（此函数可以先不开启调用，因为会有较长的数据运行）
kNN.handwritingClassTest()
