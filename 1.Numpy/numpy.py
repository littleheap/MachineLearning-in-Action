# coding=gbk

from numpy import *

# 随机数列
print(random.rand(4, 4))

# 转化矩阵
randMat = mat(random.rand(4, 4))
print(randMat)

# 矩阵求逆
invRandMat = randMat.I
print(invRandMat)

# 矩阵乘法
multi = invRandMat*randMat
print(multi)

# 矩阵减法
contrast = multi - eye(4)
print(contrast)
