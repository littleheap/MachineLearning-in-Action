from math import log
import operator
import treePlotter


# 样例数据集初始化函数
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


# 计算给定数据集的香农熵
def calcShannonEnt(dataSet):
    # 获取数据集数据组个数
    numEntries = len(dataSet)
    # 定义记录label数量字典
    labelCounts = {}
    # 遍历数据集每一条数据
    for featVec in dataSet:
        # 获取当前这条数据的label值
        currentLabel = featVec[-1]
        # 如果是新label，则在标签字典中新建对应的key，value的对应出现的次数，初始化为0
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        # 当前标签在数据集出现的次数加1
        labelCounts[currentLabel] += 1
    # 香农熵值初始化为0
    shannonEnt = 0.0
    # 遍历标签字典那种每一个key
    for key in labelCounts:
        # 计算每个标签占总标签的比例
        prob = float(labelCounts[key]) / numEntries
        # 累计计算香农熵
        shannonEnt -= prob * log(prob, 2)
        # 返回香农熵
    return shannonEnt


# 测试不同复杂度数据集香农熵
myDat, labels = createDataSet()
print(calcShannonEnt(myDat))
myDat[0][-1] = 'maybe'
print(calcShannonEnt(myDat))


# 按照特征划分数据集
def splitDataSet(dataSet, axis, value):
    # 初始化返回数据集
    retDataSet = []
    # 遍历输入数据集每一条数据
    for featVec in dataSet:
        # 如果数据集的第axis列值等于value，保留条数据，并删除第axis列数据信息
        if featVec[axis] == value:
            # 获取被降维特征前面的所有特征
            reducedFeatVec = featVec[:axis]
            # 衔接被降维特征后面的所有特征
            reducedFeatVec.extend(featVec[axis + 1:])
            # 新的降维数据加入新的返回数据集中
            retDataSet.append(reducedFeatVec)
    # 返回指定特征降维后的数据集
    return retDataSet


# 选择最好的数据划分
def chooseBestFeatureToSplit(dataSet):
    # 获取每条数据的特征数目
    numFeatures = len(dataSet[0]) - 1
    # 计算数据集的基础香农熵
    baseEntropy = calcShannonEnt(dataSet)
    # 定义最合理划分后的香农熵衰减
    bestInfoGain = 0.0
    # 最好的数据划分特征索引，初始化为-1
    bestFeature = -1
    # 遍历所有的特征划分特征可能
    for i in range(numFeatures):
        # 将每条数据的第i个特征纳入一个集合
        featList = [example[i] for example in dataSet]
        # 将第i个特征种类去重
        uniqueVals = set(featList)
        # 新香农熵
        newEntropy = 0.0
        # 遍历第i个特征所有种类，从此分叉
        for value in uniqueVals:
            # 获取按第i个特征划分后每个种类的子数据集
            subDataSet = splitDataSet(dataSet, i, value)
            # 计算当前子数据集的部分香农熵
            prob = len(subDataSet) / float(len(dataSet))
            # 累加到当前划分的总香农熵中
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 计算香农熵衰减
        infoGain = baseEntropy - newEntropy
        # 如果大于此前最合理衰减，则保留当前划分为最优划分
        if (infoGain > bestInfoGain):
            # 更新最优划分
            bestInfoGain = infoGain
            # 保存最优划分特征索引
            bestFeature = i
    # 返回最优划分特征索引
    return bestFeature


# 返回标签出现次数最多的标签
def majorityCnt(classList):
    # 定义分类标签出现次数字典
    classCount = {}
    # 遍历所有标签
    for vote in classList:
        # 如果当前标签不在字典中，则创建相应的key
        if vote not in classCount.keys(): classCount[vote] = 0
        # 当前标签的value加1
        classCount[vote] += 1
    # 将字典内部排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回出现次数最多的标签
    return sortedClassCount[0][0]


# 构建决策树
def createTree(dataSet, labels):
    # 将所有数据的标签纳入一个合集
    classList = [example[-1] for example in dataSet]
    # 如果当前所有的标签都一样，那就别无选择，结束决策，返回标签
    if classList.count(classList[0]) == len(classList):
        # 返回当前标签
        return classList[0]
    # 如果数据集被划分的只剩标签，没有特征了
    if len(dataSet[0]) == 1:
        # 就返回标签中概率最大的作为最后判定
        return majorityCnt(classList)
    # 获取当前数据集最优划分特征索引
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 获取最优分类标签的判别名词
    bestFeatLabel = labels[bestFeat]
    # 构造的决策树为当前最优划分标签判别名词，附加下一个判别名词的字典
    myTree = {bestFeatLabel: {}}
    # 删除判别名词合集中的当前已选最优标签
    del (labels[bestFeat])
    # 获取数据集中当前最优划分特征索引下的特征种类
    featValues = [example[bestFeat] for example in dataSet]
    # 将获取的特征种类去重
    uniqueVals = set(featValues)
    # 遍历所有特征种类
    for value in uniqueVals:
        # 获取子分类判别名词数据集，即去掉最优划分的那个名词剩下的数据集
        subLabels = labels[:]
        # 当前最优划分的特征key对应的value就是下一层决策树的返回值，递归求解
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
        # print(myTree[bestFeatLabel][value])
    return myTree


myDat, labels = createDataSet()
newTree = createTree(myDat, labels)
print(newTree)
print(newTree['no surfacing'][0])
print(newTree['no surfacing'][1])


# 使用决策树分类函数
def classify(inputTree, featLabels, testVec):
    # 获取当前树头结点
    firstStr = list(inputTree.keys())[0]
    # 获取头结点的分支
    secondDict = inputTree[firstStr]
    # 获取头节点特征标签索引
    featIndex = featLabels.index(firstStr)
    # 获取头节点对应的选项
    key = testVec[featIndex]
    # 获取下一层判定
    valueOfFeat = secondDict[key]
    # 如果下一层判定还是字典
    if isinstance(valueOfFeat, dict):
        # 递归进行深度搜索
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        # 否则不是字典，就是最后判定的标签
        classLabel = valueOfFeat
    # 返回最后的分类标签
    return classLabel


# 使用pickle模块存储决策树
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, "wb+")
    pickle.dump(inputTree, fw)
    fw.close()


# 读取预先存储的决策树
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)


# 打开txt数据文件
fr = open('lenses.txt')
# 将每行的数据整理成预设格式
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
# 打印整理后格式
print(lenses)
# 初始化四个特征值标签
lensesLabels = ['age', 'prescript', 'astigamatic', 'tearRate']
# 创建树结构
lensesTree = createTree(lenses, lensesLabels)
# 绘制树结构
treePlotter.createPlot(lensesTree)
# 存储树结构
# storeTree(lensesTree, 'lensesTree.txt')
