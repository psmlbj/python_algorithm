# -*- coding: cp936 -*-
"""
找一种从n维空间向一维空间映射的函数，使得n维空间中距离近的点在一维空间中距离也近
"""
from Tkinter import *
from cmath import polar
import time
import numpy as np
import matplotlib.pyplot as plt
import math

def randInt(min, max, *d):
    # 产生介于min与max之间的矩阵，维度有d指定
    return (np.random.rand(*d)) * (max - min) + min

def CreateDataset(min, max, *d):
    # 产生随机整型2*size矩阵，点的坐标扩大expand_ratio倍
    randMat = randInt(min, max, d[0], d[1])
    # 把二维矩阵转换成点的集合
    points = map(lambda x, y:(int(x), int(y)), *randMat)
    # 去重复
    points = list(set(points))
    return points

datamin = 10
datamax = 100
datanum = 6

data = CreateDataset(datamin, datamax, 2, datanum)
data = [[item[0], item[1]] for item in data]
xmin = min(data, key=lambda x:x[0])[0]
ymin = min(data, key=lambda x:x[1])[1]
xmax = max(data, key=lambda x:x[0])[0]
ymax = max(data, key=lambda x:x[1])[1]
basepoint1 = ((xmin + xmax) / 2, ymin)
basepoint2 = (xmin, (ymin + ymax) / 2)

basepoint1 = (datamax / 2, 0)
basepoint2 = (0, datamax / 2)
dr1 = map(lambda x:math.sqrt((x[0] - basepoint1[0]) ** 2 + (x[1] - basepoint1[1]) ** 2), data)
dr2 = map(lambda x:math.sqrt((x[0] - basepoint2[0]) ** 2 + (x[1] - basepoint2[1]) ** 2), data)
print data
print dr1
print dr2
for i in range(datanum):
    data[i].append(int(dr1[i] + dr2[i]))
    data[i].append(dr1[i] + dr2[i])
data.sort(key=lambda x:x[3])
print data
