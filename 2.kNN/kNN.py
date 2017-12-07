from numpy import *
import operator
from os import listdir


# 范例函数
def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# k-邻近算法 分类函数（分类目标，数据集，数据集标志位，临近筛选数目）
def classify0(inX, dataSet, labels, k):
    # 获取数据集每条数据的特征个数
    dataSetSize = dataSet.shape[0]
    # 计算每条数据和数据集内数据的距离
    # 将分类目标坐标纵向拉伸，与整个数据集对齐，做减法
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # 对结果做平方，消除负值影响
    sqDiffMat = diffMat ** 2
    # 将一类的距离结果相加，axis=1表示按行相加，axis=0表示按列相加
    sqDistances = sqDiffMat.sum(axis=1)
    # 再对每个距离开平方
    distances = sqDistances ** 0.5
    # 获取距离数组的从小到大排序索引值
    sortedDistIndicies = distances.argsort()
    # 定义分类字典
    classCount = {}
    # 选取最小的k个距离并排序
    for i in range(k):
        # 从labels获取对应类别分类
        voteIlabel = labels[sortedDistIndicies[i]]
        # 将获取到的分类累加到分类字典中，次数越多，对应的value值越大
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 对字典内部进行排序，返回副本
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回累加次数最多的类别名称
    return sortedClassCount[0][0]


# 文本记录转化为矩阵
def file2matrix(filename):
    fr = open(filename)
    # 获取文件的行数
    numberOfLines = len(fr.readlines())
    # 创建返回值矩阵，行数与文本一致，列数为3，初始化为0
    returnMat = zeros((numberOfLines, 3))
    # 创建分类向量集合，存储每条数据的labels
    classLabelVector = []
    # 解析预备数据
    index = 0
    fr = open(filename)
    for line in fr.readlines():
        # 移除每一行数据的首尾空格
        line = line.strip()
        # 将数据去间隙，压入字符串数组
        listFromLine = line.split('\t')
        # 将文本数据中，前三个特征数据送入返回矩阵对应的三列中
        returnMat[index, :] = listFromLine[0:3]
        # 将文本数据中，对应的分类类别依次送入分类向量集合
        classLabelVector.append(int(listFromLine[-1]))
        # 游标累加1，获取下一行数据
        index += 1
    # 返回特征向量矩阵和对应分类结果集合
    return returnMat, classLabelVector


# 归一化函数
def autoNorm(dataSet):
    # 参数为0，获取矩阵每列最小值
    minVals = dataSet.min(0)
    # 参数为0，获取矩阵每列最大值
    maxVals = dataSet.max(0)
    # 求出每列的数值范畴
    ranges = maxVals - minVals
    # 创建和输入矩阵尺寸一样的正规矩阵，初始化为0
    normDataSet = zeros(shape(dataSet))
    # 获取输入矩阵行数
    m = dataSet.shape[0]
    # 将最小值矩阵纵向拉伸，与输入矩阵对应，并与输入矩阵做差
    normDataSet = dataSet - tile(minVals, (m, 1))
    # 将结果再除以相应特征范畴，归一化
    normDataSet = normDataSet / tile(ranges, (m, 1))
    # 返回归一化矩阵，特征值范畴，最小特征值矩阵
    return normDataSet, ranges, minVals


# 测试函数
def datingClassTest():
    # 划分测试和基础样本比例
    hoRatio = 0.10
    # 读取数据和分类标签矩阵
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')  # load data setfrom file
    # 获取归一化数据，包括归一矩阵，各特征范畴，各特征最小值
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # 获取数据总数量
    m = normMat.shape[0]
    # 获取基础样本数量
    numTestVecs = int(m * hoRatio)
    # 初始化错误次数
    errorCount = 0.0
    print('\n测试样本(显示前20个)\n')
    for i in range(numTestVecs):
        # 依次以前90%中的每一条数据作为分类目标输入，后10%数据和标签作为分类基础数据
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        # 打印前20个结果和预测值比较
        if (i < 20): print(
            'the classifier came back with: %d, the real answer is: %d' % (classifierResult, datingLabels[i]))
        # 比对分类结果和真实结果
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print('\n错误率：%f\n' % (errorCount / float(numTestVecs)))
    print('\n错误数：%d\n' % errorCount)


# 手写数字识别系统
print('\n手写识别系统\n')

# 将二维数据图片转化为一维矩阵数据
def img2vector(filename):
    # 创建1*1024矩阵
    returnVect = zeros((1, 1024))
    # 打开文件
    fr = open(filename)
    # 将每一行每一列对应的数值归到整个一维矩阵中
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    # 返回该一维矩阵
    return returnVect


# 手写数字训练及测试函数
def handwritingClassTest():
    # 定义类别标签集合
    hwLabels = []
    # 获取训练目录数据进行训练
    trainingFileList = listdir('trainingDigits')
    # 获取训练集数量
    m = len(trainingFileList)
    # 训练集矩阵尺寸为m*1024
    trainingMat = zeros((m, 1024))
    # 从文件中解析数字数据
    for i in range(m):
        # 读取文件名
        fileNameStr = trainingFileList[i]
        # 排除txt尾缀
        fileStr = fileNameStr.split('.')[0]
        # 获取该文件对应的真实数字值
        classNumStr = int(fileStr.split('_')[0])
        # 依次加入类别标签集合
        hwLabels.append(classNumStr)
        # 将每一个图片txt文件转化为一维向量，存入训练集矩阵
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    # 获取测试集
    testFileList = listdir('testDigits')
    # 初始化错误记录
    errorCount = 0.0
    # 获取测试集数量
    mTest = len(testFileList)
    for i in range(mTest):
        # 获取每一个测试集文件
        fileNameStr = testFileList[i]
        # 排除txt尾缀
        fileStr = fileNameStr.split('.')[0]
        # 获取对应真实分类类别
        classNumStr = int(fileStr.split('_')[0])
        # 将其转化为一维矩阵
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        # 以训练集为基准，目标奖测试集中每一个数字分类，提取前三个概率最大可能
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print('the classifier came back with: %d, the real answer is: %d' % (classifierResult, classNumStr))
        # 比对预测值和真实值
        if (classifierResult != classNumStr): errorCount += 1.0
    print('\n错误个数: %d' % errorCount)
    print('\n错误率: %f' % (errorCount / float(mTest)))
