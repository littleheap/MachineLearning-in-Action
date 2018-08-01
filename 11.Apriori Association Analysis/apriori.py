from numpy import *


# 创建一个用于测试的简单数据集
def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


# C1是大小为1的所有候选项集的集合
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))


# scanD将C1生成L1
def scanD(D, Ck, minSupport):
    # 参数：数据集、候选项集列表、感兴趣项集的最小支持度
    ssCnt = {}
    # 遍历数据集
    for tid in D:
        # 遍历候选项
        for can in Ck:
            if can.issubset(tid):  # 判断候选项中是否含数据集的各项
                if not can in ssCnt:
                    # 不含设为1
                    ssCnt[can] = 1
                else:
                    # 有则计数加1
                    ssCnt[can] += 1
                    # 数据集大小
    numItems = float(len(D))
    # L1初始化
    retList = []
    # 记录候选项中各个数据的支持度
    supportData = {}
    for key in ssCnt:
        # 计算支持度
        support = ssCnt[key] / numItems
        if support >= minSupport:
            # 满足条件加入L1中
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData


dataset = loadDataSet()

print(dataset)

# >>>[[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

C1 = createC1(dataset)

print(C1)

# >>>[frozenset({1}), frozenset({2}), frozenset({3}), frozenset({4}), frozenset({5})]

D = list(map(set, dataset))

print(D)

# >>>[{1, 3, 4}, {2, 3, 5}, {1, 2, 3, 5}, {2, 5}]

L1, supportData0 = scanD(D, C1, 0.5)

print(L1)


# >>>[frozenset({5}), frozenset({2}), frozenset({3}), frozenset({1})]

# 完整Apriori算法
def aprioriGen(Lk, k):
    # 参数：频繁项集列表、项集元素个数
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        # 两两组合遍历
        for j in range(i + 1, lenLk):
            L1 = list(Lk[i])[:k - 2]
            L2 = list(Lk[j])[:k - 2]
            L1.sort()
            L2.sort()
            # 若两个集合的前k-2个项相同时,则将两个集合合并
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList


# 给定该数据集和支持度，函数生成候选项列表
def apriori(dataSet, minSupport=0.5):
    C1 = createC1(dataSet)
    D = list(map(set, dataSet))
    # 单项最小支持度判断 0.5，生成L1
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    # 创建包含更大项集的更大列表,直到下一个大的项集为空
    while (len(L[k - 2]) > 0):
        Ck = aprioriGen(L[k - 2], k)  # Ck
        Lk, supK = scanD(D, Ck, minSupport)  # Lk
        supportData.update(supK)
        L.append(Lk)
        # 继续向上合并生成项集个数更多的
        k += 1
    return L, supportData


L, suppData = apriori(dataset)

print(L)

# >>>[[frozenset({5}), frozenset({2}), frozenset({3}), frozenset({1})], [frozenset({2, 3}), frozenset({3, 5}), frozenset({2, 5}), frozenset({1, 3})], [frozenset({2, 3, 5})], []]

print(L[0])

# >>>[frozenset({5}), frozenset({2}), frozenset({3}), frozenset({1})]

print(L[1])

# >>>[frozenset({2, 3}), frozenset({3, 5}), frozenset({2, 5}), frozenset({1, 3})]

print(L[2])

# >>>[frozenset({2, 3, 5})]

print(aprioriGen(L[0], 2))

# >>>[frozenset({2, 5}), frozenset({3, 5}), frozenset({1, 5}), frozenset({2, 3}), frozenset({1, 2}), frozenset({1, 3})]

L, suppData = apriori(dataset, 0.7)

print(L)


# >>>[[frozenset({5}), frozenset({2}), frozenset({3})], [frozenset({2, 5})], []]


# 生成关联规则
def generateRules(L, supportData, minConf=0.7):
    # 频繁项集列表、包含那些频繁项集支持数据的字典、最小可信度阈值
    bigRuleList = []  # 存储所有的关联规则
    for i in range(1, len(L)):  # 只获取有两个或者更多集合的项目，从1,即第二个元素开始，L[0]是单个元素的
        # 两个及以上的才可能有关联一说，单个元素的项集不存在关联问题
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            # 该函数遍历L中的每一个频繁项集并对每个频繁项集创建只包含单个元素集合的列表H1
            if (i > 1):
                # 如果频繁项集元素数目超过2,那么会考虑对它做进一步的合并
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:  # 第一层时，后件数为1
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)  # 调用函数2
    return bigRuleList


# 生成候选规则集合：计算规则的可信度以及找到满足最小可信度要求的规则
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    # 针对项集中只有两个元素时，计算可信度
    # 返回一个满足最小可信度要求的规则列表
    prunedH = []
    # 遍历H中的所有项集并计算它们的可信度值
    for conseq in H:
        # 可信度计算，结合支持度数据
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        if conf >= minConf:
            # 如果某条规则满足最小可信度值,那么将这些规则输出到屏幕显示
            print(freqSet - conseq, '-->', conseq, 'conf:', conf)
            # 添加到规则里，brl是前面通过检查的bigRuleList
            brl.append((freqSet - conseq, conseq, conf))
            # 同样需要放入列表到后面检查
            prunedH.append(conseq)
    return prunedH


# 合并
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    # 参数:一个是频繁项集,另一个是可以出现在规则右部的元素列表H
    m = len(H[0])
    # 频繁项集元素数目大于单个集合的元素数
    if (len(freqSet) > (m + 1)):
        # 存在不同顺、元素相同的集合，合并具有相同部分的集合
        Hmp1 = aprioriGen(H, m + 1)
        # 计算可信度
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):
            # 满足最小可信度要求的规则列表多于1,则递归来判断是否可以进一步组合这些规则
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)


L, suppData = apriori(dataset, 0.5)

print(L)

# >>>[[frozenset({5}), frozenset({2}), frozenset({3}), frozenset({1})], [frozenset({2, 3}), frozenset({3, 5}), frozenset({2, 5}), frozenset({1, 3})], [frozenset({2, 3, 5})], []]

print(suppData)

# >>>{frozenset({1}): 0.5, frozenset({3}): 0.75, frozenset({4}): 0.25, frozenset({2}): 0.75, frozenset({5}): 0.75, frozenset({1, 3}): 0.5, frozenset({2, 5}): 0.75, frozenset({3, 5}): 0.5, frozenset({2, 3}): 0.5, frozenset({1, 5}): 0.25, frozenset({1, 2}): 0.25, frozenset({2, 3, 5}): 0.5}

rules = generateRules(L, suppData, 0.7)

print(rules)

'''
    frozenset({5}) --> frozenset({2}) conf: 1.0
    frozenset({2}) --> frozenset({5}) conf: 1.0
    frozenset({1}) --> frozenset({3}) conf: 1.0
    [(frozenset({5}), frozenset({2}), 1.0), (frozenset({2}), frozenset({5}), 1.0), (frozenset({1}), frozenset({3}), 1.0)]
'''
