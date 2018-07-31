from numpy import *
import matplotlib.pyplot as plt


# 数据加载方法
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # 数值转换为浮点类型
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat


# 向量欧式距离计算方法
def distEclud(vecA, vecB):
    # 向量AB的欧式距离
    return sqrt(sum(power(vecA - vecB, 2)))


# 为给定的数据集初始化K个随机质心集合
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))
    for j in range(n):
        # 寻找每一维度数据最大值最小值
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        # 确保随机点在数据边界之内
        centroids[:, j] = mat(minJ + rangeJ * random.rand(k, 1))
    return centroids


# K-Means算法
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            # 寻找最近的质心
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                # 此处判断数据点所属类别与之前是否相同（是否变化，只要有一个点变化就重设为True，再次迭代）
            clusterAssment[i, :] = minIndex, minDist ** 2
        # 打印每次迭代后质心的结果
        print(centroids)
        # 更新质心的位置
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment


# 二分K-均值聚类
def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    # 创建一个初始簇
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0]
    for j in range(m):
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j, :]) ** 2
    while (len(centList) < k):
        lowestSSE = inf
        # 尝试划分每一簇
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:, 1])
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print("sseSplit, and notSplit: ", sseSplit, sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        # 跟新簇的分配结果
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print('the bestCentToSplit is: ', bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]
        centList.append(bestNewCents[1, :].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
    return mat(centList), clusterAssment


# 球面距离计算及簇绘图函数
def distSLC(vecA, vecB):
    a = sin(vecA[0, 1] * pi / 180) * sin(vecB[0, 1] * pi / 180)
    b = cos(vecA[0, 1] * pi / 180) * cos(vecB[0, 1] * pi / 180) * cos(pi * (vecB[0, 0] - vecA[0, 0]) / 180)
    return arccos(a + b) * 6371.0


def clusterClubs(numClust=5):  # numClust：希望得到的簇数目
    datList = []
    # 获取地图数据
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        # 逐个获取第四列和第五列的经纬度信息
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    # 绘图
    fig = plt.figure()
    # 创建矩形
    rect = [0.1, 0.1, 0.8, 0.8]
    # 创建不同标记图案
    scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')  # 导入地图
    ax0.imshow(imgP)
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:, 0].A == i)[0], :]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0], ptsInCurrCluster[:, 1].flatten().A[0], marker=markerStyle,
                    s=90)
    ax1.scatter(myCentroids[:, 0].flatten().A[0], myCentroids[:, 1].flatten().A[0], marker='+', s=300)
    plt.show()


dataMat1 = mat(loadDataSet('testSet.txt'))

print(min(dataMat1[:, 0]))  # [[4.838138]]

print(max(dataMat1[:, 0]))  # [[4.838138]]

print(min(dataMat1[:, 1]))  # [[-4.232586]]

print(max(dataMat1[:, 1]))  # [[5.1904]]

print(randCent(dataMat1, 2))

'''
    [[ 2.04563708 -2.76511894]
    [-3.28512825 -0.0110467 ]]
'''

dataMat2 = mat(loadDataSet('testSet.txt'))

myCentroids, clustAssing = kMeans(dataMat2, 4)

print(myCentroids)  # 打印4个质心

'''
    [[-3.38237045 -2.9473363 ]
     [-2.46154315  2.78737555]
     [ 2.6265299   3.10868015]
     [ 2.80293085 -2.7315146 ]]
'''

dataMat3 = mat(loadDataSet('testSet2.txt'))

centList, myNewAssments = biKmeans(dataMat3, 3)

print(centList)  # 打印3个质心结果

'''
    [[-0.45965615 -2.7782156 ]
     [-2.94737575  3.3263781 ]
     [ 2.93386365  3.12782785]]

'''

clusterClubs(4)
