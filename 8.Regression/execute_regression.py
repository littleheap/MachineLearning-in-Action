from numpy import *
import matplotlib.pyplot as plt
from Regression import regression

xArr, yArr = regression.loadDataSet('ex0.txt')

print(xArr[0:10])
print(yArr[0:10])

ws = regression.standRegres(xArr, yArr)
print(ws)

xMat = mat(xArr)
yMat = mat(yArr)
yHat = xMat * ws

# 图像显示
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])

xCopy = xMat.copy()
xCopy.sort(0)
yHat = xCopy * ws
ax.plot(xCopy[:, 1], yHat)
plt.show()

# 计算拟合率
yHat = xMat * ws
print(corrcoef(yHat.T, yMat))

# 局部加权线性回归
xArr, yArr = regression.loadDataSet('ex0.txt')
print(xArr[0])
regression.lwlr(xArr[0], xArr, yArr, 1.0)
yHat = regression.lwlrTest(xArr, xArr, yArr, 0.02)  # 通过调节最后一个系数调节局部权值

strInd = xMat[:, 1].argsort(0)
xSort = xMat[strInd][:, 0, :]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xSort[:, 1], yHat[strInd])
ax.scatter(xMat[:, 1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2, c='red')
plt.show()

# 岭回归
abX, abY = regression.loadDataSet('abalone.txt')
ridgeWeights = regression.ridgeTest(abX, abY)
# 显示岭回归图像，通过图像选取合适lambda值
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(ridgeWeights)
plt.show()

# 向前逐步回归
xArr, yArr = regression.loadDataSet('abalone.txt')
final = regression.stageWise(xArr, yArr, 0.01, 300)
# 显示回归系数变化情况图像
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(final)
plt.show()
