from numpy import *


# 数据集初始化函数
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 6个句子，对应列表6个01元素，1代表有侮辱性文字，0代表没有侮辱性文字
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


# 创建数据集集合列表，将出现的词合并入一个集合，返回列表形式
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        # 不断取并集
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


# 词表到向量的转换函数
# 将出现文字和预先词汇表比对，出现则对应索引鸡1，否则默认为0
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print
        "the word: %s is not in my Vocabulary!" % word
    return returnVec


listOPosts, listClasses = loadDataSet()
myVocabList = createVocabList(listOPosts)
print(myVocabList)

v0 = setOfWords2Vec(myVocabList, listOPosts[0])
print(v0)
v3 = setOfWords2Vec(myVocabList, listOPosts[3])
print(v3)


# 训练算法
# [训练矩阵：由6个句子的向量组合成的总向量；训练分类：即数据集label矩阵]
def trainNB0(trainMatrix, trainCategory):
    # 获取训练矩阵长度，即多少个句子，此处为6
    numTrainDocs = len(trainMatrix)
    # 获取库中所有单词数量，也就是矩阵中特征数量
    numWords = len(trainMatrix[0])
    # 计算侮辱性句子总体占比，此处为1/2
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 初始化概率
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    # p0Num = zeros(numWords)
    # p1Num = zeros(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    # p0Denom = 0
    # p1Denom = 0
    # 遍历所有句子向量
    for i in range(numTrainDocs):
        # 如果当前句子为侮辱性的
        if trainCategory[i] == 1:
            # 累加句子向量
            p1Num += trainMatrix[i]
            # 累加句中词库中词出现的数量
            p1Denom += sum(trainMatrix[i])
        else:
            # 如但如果当前句子为非侮辱性的正常言论
            # 累加句子向量
            p0Num += trainMatrix[i]
            # 累加句中词库中词出现的数量
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)
    # 对每个元素做除法
    # p1Vect = p1Num / p1Denom
    # p0Vect = p0Num / p0Denom
    # 返回两种类型的算术矩阵和侮辱言论占比
    return p0Vect, p1Vect, pAbusive


listOPosts, listClasses = loadDataSet()
myVocabList = createVocabList(listOPosts)
trainMat = []
for post in listOPosts:
    trainMat.append(setOfWords2Vec(myVocabList, post))
print(trainMat)
p0V, p1V, pAb = trainNB0(trainMat, listClasses)
print(p0V)
print(p1V)
print(pAb)
print('\n')


# 朴素贝叶斯分类函数
# [测试输入语义向量，正常评论词汇概率矩阵，侮辱言论词汇概率矩阵，侮辱言论数据集句子概率]
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # 侮辱性预判值
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    print('p1:', p1)
    # 正常言论预判值
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    print('p0:', p0)
    # 比较预判值
    if p1 > p0:
        return 1
    else:
        return 0


# 测试函数
def testingNB():
    # 获取数据集中句子和分类
    listOPosts, listClasses = loadDataSet()
    # 生成词库
    myVocabList = createVocabList(listOPosts)
    # 初始化语义向量矩阵
    trainMat = []
    # 遍历所有句子数据，生成总体语义矩阵
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    # 求出正常评论词汇概率矩阵，侮辱言论词汇概率矩阵，和侮辱言论数据集句子概率
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    # 测试输入
    testEntry = ['love', 'my', 'dalmation']
    # 转换成对应语义向量
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    # 打印分类情况
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    # 测试输入
    testEntry = ['stupid', 'garbage']
    # 转换成对应语义向量
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    # 打印分类情况
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))


testingNB()
print('\n')


# 朴素贝叶斯词袋模型，将之前的词集合统计换成累加
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


# 文本转型，去除少于2个字符的字符串，转小写
def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


# 测试分类函数
def spamTest():
    # 文件信息矩阵
    docList = []
    # 分类矩阵
    classList = []
    # 全部文本信息矩阵
    fullText = []
    # 循环25次读取两个文件夹中50个邮件文本
    for i in range(1, 26):
        # 处理1类文本信息
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        # 衔接到文件信息矩阵
        docList.append(wordList)
        # 拓展到全部文本信息矩阵
        fullText.extend(wordList)
        # 分类信息衔接1类标志
        classList.append(1)
        # 处理0类文本信息
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        # 衔接到文件信息矩阵
        docList.append(wordList)
        # 拓展到全部文本信息矩阵
        fullText.extend(wordList)
        # 分类信息衔接0类标志
        classList.append(0)
    # 依据收集的文本信息，创建词汇集合
    vocabList = createVocabList(docList)
    # 初始化训练集合索引记录1-50
    trainingSet = list(range(50))
    # 初始测试矩阵
    testSet = []
    # 随机选取十个信息数据作为测试集
    for i in range(10):
        # 1-50随机选取一个数字
        randIndex = int(random.uniform(0, len(trainingSet)))
        # 将选取到的衔接到测试集索引区
        testSet.append(trainingSet[randIndex])
        # 删除训练集对应索引
        del (trainingSet[randIndex])
    # 初始化训练矩阵
    trainMat = []
    # 初始化分类矩阵
    trainClasses = []
    # 遍历训练集索引，此处有40个
    for docIndex in trainingSet:
        # 构造训练集矩阵，依据词袋函数
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        # 构造训练集对应的分类矩阵
        trainClasses.append(classList[docIndex])
    # 计算0分类语义概率，1分类语义概率，1分类概率
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    # 初始化预测错误次数
    errorCount = 0
    # 遍历测试集，此处有10个
    for docIndex in testSet:
        # 构造词袋向量
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        # 判定预测与实际值异同
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            # 不一样累加1
            errorCount += 1
            print("classification error", docList[docIndex])
    # 计算并打印预测错误率
    print('the error rate is: ', float(errorCount) / len(testSet))


spamTest()


def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]


def localWords(feed1, feed0):
    import feedparser
    docList = [];
    classList = [];
    fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)  # NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)  # create vocabulary
    top30Words = calcMostFreq(vocabList, fullText)  # remove top 30 words
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = range(2 * minLen);
    testSet = []  # create test set
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = [];
    trainClasses = []
    for docIndex in trainingSet:  # train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:  # classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print
    'the error rate is: ', float(errorCount) / len(testSet)
    return vocabList, p0V, p1V


def getTopWords(ny, sf):
    import operator
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = [];
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0: topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0: topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print
    "SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**"
    for item in sortedSF:
        print
        item[0]
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print
    "NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**"
    for item in sortedNY:
        print
        item[0]
