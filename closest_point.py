# -*- coding: cp936 -*-
"""
��һ�ִ�nά�ռ���һά�ռ�ӳ��ĺ�����ʹ��nά�ռ��о�����ĵ���һά�ռ��о���Ҳ��
"""
from Tkinter import *
from cmath import polar
import time
import numpy as np
import matplotlib.pyplot as plt
import math

def randInt(min, max, *d):
    # ��������min��max֮��ľ���ά����dָ��
    return (np.random.rand(*d)) * (max - min) + min

def CreateDataset(min, max, *d):
    # �����������2*size���󣬵����������expand_ratio��
    randMat = randInt(min, max, d[0], d[1])
    # �Ѷ�ά����ת���ɵ�ļ���
    points = map(lambda x, y:(int(x), int(y)), *randMat)
    # ȥ�ظ�
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
