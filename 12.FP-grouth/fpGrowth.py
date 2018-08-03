import twitter
from time import sleep
import re


# FP树中节点的类定义
class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        # nodeLink 变量用于链接相似的元素项
        self.nodeLink = None
        # 指向当前节点的父节点
        self.parent = parentNode
        # 空字典，存放节点的子节点
        self.children = {}

    def inc(self, numOccur):
        self.count += numOccur

    # 将树以文本形式显示
    def disp(self, ind=1):
        print('  ' * ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind + 1)


# 构建FP-tree
def createTree(dataSet, minSup=1):
    headerTable = {}
    # 第一次遍历：统计各个数据的频繁度
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
            # 用头指针表统计各个类别的出现的次数，计算频繁量：头指针表[类别]=出现次数
    # 删除未达到最小频繁度的数据
    for k in list(headerTable.keys()):
        if headerTable[k] < minSup:
            del (headerTable[k])
    # 保存达到要求的数据
    freqItemSet = set(headerTable.keys())
    # print ('freqItemSet: ',freqItemSet)
    if len(freqItemSet) == 0:
        # 若达到要求的数目为0
        return None, None
        # 遍历头指针表
    for k in headerTable:
        # 保存计数值及指向每种类型第一个元素项的指针
        headerTable[k] = [headerTable[k], None]
    # print ('headerTable: ',headerTable)
    # 初始化tree
    retTree = treeNode('Null Set', 1, None)
    # 第二次遍历：
    for tranSet, count in dataSet.items():
        localD = {}
        for item in tranSet:
            # 只对频繁项集进行排序
            if item in freqItemSet:
                localD[item] = headerTable[item][0]

        # 使用排序后的频率项集对树进行填充
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            updateTree(orderedItems, retTree, headerTable, count)
            # 返回树和头指针表
    return retTree, headerTable


def updateTree(items, inTree, headerTable, count):
    # 首先检查是否存在该节点
    if items[0] in inTree.children:
        # 存在则计数增加
        inTree.children[items[0]].inc(count)
    else:  # 不存在则将新建该节点
        # 创建一个新节点
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        # 若原来不存在该类别，更新头指针列表
        if headerTable[items[0]][1] == None:
            # 更新指向
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            # 更新指向
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    # 仍有未分配完的树，迭代
    if len(items) > 1:
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)


# 节点链接指向树中该元素项的每一个实例。
# 从头指针表的 nodeLink 开始,一直沿着nodeLink直到到达链表末尾
def updateHeader(nodeToTest, targetNode):
    while (nodeToTest.nodeLink != None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


# 从FP树中发现频繁项集
# 递归上溯整棵树
def ascendTree(leafNode, prefixPath):
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)


def findPrefixPath(basePat, treeNode):  # 参数：指针，节点；
    condPats = {}
    while treeNode != None:
        prefixPath = []
        # 寻找当前非空节点的前缀
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            # 将条件模式基添加到字典中
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats


# 递归查找频繁项集
def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    # 头指针表中的元素项按照频繁度排序,从小到大
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: str(p[1]))]
    # 从底层开始
    for basePat in bigL:
        # 加入频繁项列表
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        print('finalFrequent Item: ', newFreqSet)
        freqItemList.append(newFreqSet)
        # 递归调用函数来创建基
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        print('condPattBases :', basePat, condPattBases)

        # 2. 构建条件模式Tree
        myCondTree, myHead = createTree(condPattBases, minSup)
        # 将创建的条件基作为新的数据集添加到fp-tree
        print('head from conditional tree: ', myHead)
        if myHead != None:  # 3. 递归
            print('conditional tree for: ', newFreqSet)
            myCondTree.disp(1)
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)


def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat


# createInitSet() 用于实现上述从列表到字典的类型转换过程
def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict


def textParse(bigString):
    urlsRemoved = re.sub('(http:[/][/]|www.)([a-z]|[A-Z]|[0-9]|[/.]|[~])*', '', bigString)
    listOfTokens = re.split(r'\W*', urlsRemoved)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def getLotsOfTweets(searchStr):
    CONSUMER_KEY = ''
    CONSUMER_SECRET = ''
    ACCESS_TOKEN_KEY = ''
    ACCESS_TOKEN_SECRET = ''
    api = twitter.Api(consumer_key=CONSUMER_KEY, consumer_secret=CONSUMER_SECRET,
                      access_token_key=ACCESS_TOKEN_KEY,
                      access_token_secret=ACCESS_TOKEN_SECRET)
    # you can get 1500 results 15 pages * 100 per page
    resultsPages = []
    for i in range(1, 15):
        print("fetching page %d" % i)
        searchResults = api.GetSearch(searchStr, per_page=100, page=i)
        resultsPages.append(searchResults)
        sleep(6)
    return resultsPages


def mineTweets(tweetArr, minSup=5):
    parsedList = []
    for i in range(14):
        for j in range(100):
            parsedList.append(textParse(tweetArr[i][j].text))
    initSet = createInitSet(parsedList)
    myFPtree, myHeaderTab = createTree(initSet, minSup)
    myFreqList = []
    mineTree(myFPtree, myHeaderTab, minSup, set([]), myFreqList)
    return myFreqList


# 将数据集导入
parsedDat = [line.split() for line in open('kosarak.dat').readlines()]

# 对初始集合格式化
initSet = createInitSet(parsedDat)

# 构建FP树,并从中寻找那些至少被10万人浏览过的新闻报道
myFPtree, myHeaderTab = createTree(initSet, 100000)

# 需要创建一个空列表来保存这些频繁项集
myFreqList = []
mineTree(myFPtree, myHeaderTab, 100000, set([]), myFreqList)

# 看下有多少新闻报道或报道集合曾经被10万或者更多的人浏览过:
print(len(myFreqList))  # >>>9

# 看看具体为哪9项：
print(myFreqList)

# >>>[{'1'}, {'1', '6'}, {'3'}, {'11', '3'}, {'11', '3', '6'}, {'3', '6'}, {'11'}, {'11', '6'}, {'6'}]
