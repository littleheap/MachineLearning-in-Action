from numpy import *
import matplotlib.pyplot as plt


def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    # 使用两个list来构建矩阵
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float, line)) for line in stringArr]
    return mat(datArr)


# topNfeat为可选参数，记录特征值个数
def pca(dataMat, topNfeat=9999999):
    # 求均值
    meanVals = mean(dataMat, axis=0)
    # 归一化数据
    meanRemoved = dataMat - meanVals
    # 求协方差
    covMat = cov(meanRemoved, rowvar=0)
    # 计算特征值和特征向量
    eigVals, eigVects = linalg.eig(mat(covMat))
    # 对特征值进行排序，默认从小到大
    eigValInd = argsort(eigVals)
    # 逆序取得特征值最大的元素
    eigValInd = eigValInd[:-(topNfeat + 1):-1]
    # 用特征向量构成矩阵
    redEigVects = eigVects[:, eigValInd]
    # 用归一化后的各个数据与特征矩阵相乘，映射到新的空间
    lowDDataMat = meanRemoved * redEigVects
    # 还原原始数据
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat


def replaceNanWithMean():
    datMat = loadDataSet('secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:, i].A))[0], i])
        datMat[nonzero(isnan(datMat[:, i].A))[0], i] = meanVal
    return datMat


dataMat = loadDataSet('testSet.txt')

lowDMat, reconMat = pca(dataMat, 1)

print(shape(lowDMat))  # >>>(1000, 1)

# 二维降维到一维
fig = plt.figure()

ax = fig.add_subplot(111)

ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], marker='^', s=90)

ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker='o', s=50, c='red')

fig.show()

# 与原数据重合
lowDMat, reconMat = pca(dataMat, 2)

fig = plt.figure()

ax = fig.add_subplot(111)

ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], marker='^', s=90)

ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker='o', s=50, c='red')

fig.show()
