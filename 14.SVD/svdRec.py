from numpy import *
from numpy import linalg as la


def loadExData():
    return [[0, 0, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 1, 0, 0]]


Data = loadExData()

U, Sigma, VT = la.svd(Data)

print(Sigma)  # [9.64365076e+00 5.29150262e+00 7.40623935e-16 4.05103551e-16 2.21838243e-32]

Sig3 = mat([[Sigma[0], 0, 0], [0, Sigma[1], 0], [0, 0, Sigma[2]]])

newData = U[:, :3] * Sig3 * VT[:3, :]

print(newData.shape)  # (7, 5)


def loadExData2():
    return [[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
            [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
            [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
            [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
            [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
            [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
            [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
            [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
            [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
            [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]


# 欧式距离相似度
def ecludSim(inA, inB):
    return 1.0 / (1.0 + la.norm(inA - inB))


# 皮尔逊相关系数
def pearsSim(inA, inB):
    if len(inA) < 3: return 1.0
    return 0.5 + 0.5 * corrcoef(inA, inB, rowvar=0)[0][1]


# 余弦相似度
def cosSim(inA, inB):
    num = float(inA.T * inB)
    denom = la.norm(inA) * la.norm(inB)
    return 0.5 + 0.5 * (num / denom)


myMat = mat(loadExData())

print(myMat)

'''
    [[0 0 0 2 2]
     [0 0 0 3 3]
     [0 0 0 1 1]
     [1 1 1 0 0]
     [2 2 2 0 0]
     [5 5 5 0 0]
     [1 1 1 0 0]]
'''

print(ecludSim(myMat[:, 0], myMat[:, 4]))  # 0.12973190755680383

print(ecludSim(myMat[:, 3], myMat[:, 4]))  # 1.0

print(cosSim(myMat[:, 0], myMat[:, 4]))  # 0.5

print(cosSim(myMat[:, 3], myMat[:, 4]))  # 1.0

print(pearsSim(myMat[:, 0], myMat[:, 4]))  # 0.20596538173840329

print(pearsSim(myMat[:, 3], myMat[:, 4]))  # 0.9999999999999999


# 估计评分函数：数据矩阵、用户编号、相似度计算方法和物品编号
def standEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0;
    ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0: continue
        # 寻找两个用户都做了评价的产品
        overLap = nonzero(logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))[0]
        if len(overLap) == 0:
            similarity = 0
        else:
            # 存在两个用户都评价的产品 计算相似度
            similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        # 计算每个用户对所有评价产品累计相似度
        simTotal += similarity
        # 根据评分计算比率
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal


# 推荐实现：recommend() 产生了最高的 N 个推荐结果
def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    # 寻找用户未评价的产品
    unratedItems = nonzero(dataMat[user, :].A == 0)[1]
    if len(unratedItems) == 0: return ('you rated everything')
    itemScores = []
    for item in unratedItems:
        # 基于相似度的评分
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]


myMat = mat(loadExData())

myMat[0, 1] = myMat[0, 0] = myMat[1, 0] = myMat[2, 0] = 4

myMat[3, 3] = 2

print(myMat)

'''
    [[4 4 0 2 2]
     [4 0 0 3 3]
     [4 0 0 1 1]
     [1 1 1 2 0]
     [2 2 2 0 0]
     [5 5 5 0 0]
     [1 1 1 0 0]]
'''

print(recommend(myMat, 2))

'''
    the 1 and 0 similarity is: 1.000000
    the 1 and 3 similarity is: 0.928746
    the 1 and 4 similarity is: 1.000000
    the 2 and 0 similarity is: 1.000000
    the 2 and 3 similarity is: 1.000000
    the 2 and 4 similarity is: 0.000000
    [(2, 2.5), (1, 2.0243290220056256)]
'''

print(recommend(myMat, 2, simMeas=ecludSim))

'''
    the 1 and 0 similarity is: 1.000000
    the 1 and 3 similarity is: 0.309017
    the 1 and 4 similarity is: 0.333333
    the 2 and 0 similarity is: 1.000000
    the 2 and 3 similarity is: 0.500000
    the 2 and 4 similarity is: 0.000000
    [(2, 3.0), (1, 2.8266504712098603)]
'''

print(recommend(myMat, 2, simMeas=pearsSim))

'''
    the 1 and 0 similarity is: 1.000000
    the 1 and 3 similarity is: 1.000000
    the 1 and 4 similarity is: 1.000000
    the 2 and 0 similarity is: 1.000000
    the 2 and 3 similarity is: 1.000000
    the 2 and 4 similarity is: 0.000000
    [(2, 2.5), (1, 2.0)]
'''


# 利用SVD
def svdEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0;
    ratSimTotal = 0.0
    # 不同于stanEst函数，加入了SVD分解
    U, Sigma, VT = la.svd(dataMat)
    Sig4 = mat(eye(4) * Sigma[:4])  # 建立对角矩阵
    # 降维：变换到低维空间
    xformedItems = dataMat.T * U[:, :4] * Sig4.I
    # 下面依然是计算相似度，给出归一化评分
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item:
            continue
        similarity = simMeas(xformedItems[item, :].T, xformedItems[j, :].T)
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal


U, Sigma, VT = la.svd(mat(loadExData2()))

print(Sigma)

# 计算平方和
Sig2 = Sigma ** 2

print(sum(Sig2))  # 541.9999999999995

# 取前90%
print(sum(Sig2) * 0.9)  # 487.7999999999996

# >90% SVD取前三个特征值
print(sum(Sig2[:3]))  # 500.50028912757926

print(recommend(myMat, 1, estMethod=svdEst))

'''
    the 1 and 0 similarity is: 0.498142
    the 1 and 3 similarity is: 0.498131
    the 1 and 4 similarity is: 0.509974
    the 2 and 0 similarity is: 0.552670
    the 2 and 3 similarity is: 0.552976
    the 2 and 4 similarity is: 0.217301
    [(2, 3.4177569186592387), (1, 3.330717154558564)]
'''


# 实例：SVD实现图像压缩

# 打印矩阵：由于矩阵包含了浮点数,因此必须定义浅色和深色。
def printMat(inMat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i, k]) > thresh:
                print(1,end='')
            else:
                print(0,end='')
        print(' ')


# 压缩
def imgCompress(numSV=3, thresh=0.8):
    myl = []
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = mat(myl)
    print("****original matrix******")
    printMat(myMat, thresh)
    # SVD分解得到特征矩阵
    U, Sigma, VT = la.svd(myMat)
    # 初始化新对角矩阵
    SigRecon = mat(zeros((numSV, numSV)))
    # 构造对角矩阵，将特征值填充到对角线
    for k in range(numSV):
        SigRecon[k, k] = Sigma[k]
    # 降维
    reconMat = U[:, :numSV] * SigRecon * VT[:numSV, :]
    print("****reconstructed matrix using %d singular values******" % numSV)
    printMat(reconMat, thresh)

imgCompress(2)

'''
    ****original matrix******
    00000000000000110000000000000000 
    00000000000011111100000000000000 
    00000000000111111110000000000000 
    00000000001111111111000000000000 
    00000000111111111111100000000000 
    00000001111111111111110000000000 
    00000000111111111111111000000000 
    00000000111111100001111100000000 
    00000001111111000001111100000000 
    00000011111100000000111100000000 
    00000011111100000000111110000000 
    00000011111100000000011110000000 
    00000011111100000000011110000000 
    00000001111110000000001111000000 
    00000011111110000000001111000000 
    00000011111100000000001111000000 
    00000001111100000000001111000000 
    00000011111100000000001111000000 
    00000001111100000000001111000000 
    00000001111100000000011111000000 
    00000000111110000000001111100000 
    00000000111110000000001111100000 
    00000000111110000000001111100000 
    00000000111110000000011111000000 
    00000000111110000000111111000000 
    00000000111111000001111110000000 
    00000000011111111111111110000000 
    00000000001111111111111110000000 
    00000000001111111111111110000000 
    00000000000111111111111000000000 
    00000000000011111111110000000000 
    00000000000000111111000000000000 
    ****reconstructed matrix using 2 singular values******
    00000000000000000000000000000000 
    00000000000000000000000000000000 
    00000000000001111100000000000000 
    00000000000011111111000000000000 
    00000000000111111111100000000000 
    00000000001111111111110000000000 
    00000000001111111111110000000000 
    00000000011110000000001000000000 
    00000000111100000000001100000000 
    00000000111100000000001110000000 
    00000000111100000000001110000000 
    00000000111100000000001110000000 
    00000000111100000000001110000000 
    00000000111100000000001110000000 
    00000000111100000000001110000000 
    00000000111100000000001110000000 
    00000000111100000000001110000000 
    00000000111100000000001110000000 
    00000000111100000000001110000000 
    00000000111100000000001110000000 
    00000000111100000000001110000000 
    00000000111100000000001110000000 
    00000000111100000000001110000000 
    00000000111100000000001110000000 
    00000000111100000000001110000000 
    00000000111100000000001100000000 
    00000000001111111111111000000000 
    00000000001111111111110000000000 
    00000000001111111111110000000000 
    00000000000011111111100000000000 
    00000000000011111111000000000000 
    00000000000000000000000000000000 
'''