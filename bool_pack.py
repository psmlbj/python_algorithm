# -*- coding: cp936 -*-

from Tkinter import *
import time
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import copy

class App:
    def __init__(self, master):
        # 划分区域
        self.Frame0 = Frame(master)
        self.Frame1 = Frame(master)
        self.Frame0.pack(side=TOP)
        self.Frame1.pack(side=LEFT)
        
        label = Label(self.Frame0, text=u'01背包算法')
        label.pack(side=TOP)
        self.Frame01 = Frame(self.Frame0)
        self.Frame01.pack(side=LEFT)
        
        # 设置参数
        label = Label(self.Frame01, text=u'w,C:')
        label.pack(side=LEFT)
        self.wCSV = IntVar()
        self.wCEntry = Entry(self.Frame01, textvariable=self.wCSV)
        # ,state='readonly'
        self.wCSV.set('100')
        self.wCEntry.pack(side=LEFT)

        
        label = Label(self.Frame01, text=u'e:')
        label.pack(side=LEFT)
        self.eSV = StringVar()
        self.eEntry = Entry(self.Frame01, textvariable=self.eSV)
        self.eSV.set('0.1')
        self.eEntry.pack(side=LEFT)
        
        label = Label(self.Frame01, text=u'n:')
        label.pack(side=LEFT)
        self.sizeSV = IntVar()
        self.sizeEntry = Entry(self.Frame01, textvariable=self.sizeSV)
        self.sizeSV.set('10')
        self.sizeEntry.pack(side=LEFT)
        
        self.CompareBt = Button(self.Frame01, text=u"执行", command=self.OneTime)
        self.CompareBt.pack(side=LEFT)
        
        
        label = Label(self.Frame01, text=u'               ')
        label.pack(side=LEFT)
        
        
        self.CompareBt = Button(self.Frame01, text=u"比较两种算法", command=self.Compare)
        self.CompareBt.pack(side=LEFT)
        

        
    def Compare(self):
        self.times = []
        x = []
        for n in [10, 100, 1000]:
            for wC in [100, 1000, 10000]:
                x.append((n, wC))
                Dataset = self.CreateDataset(wC, n)
                time0 = time.clock()
                self.DynamicProgramingBP(*Dataset)
                self.times.append(time.clock() - time0)
        plt.figure(figsize=(10, 7))
        plt.plot(range(1, 10), self.times, label=r"$DP$", color="red", linewidth=2)
        self.times = []
        x = []
        for n in [10, 100, 1000]:
            for wC in [100, 1000, 10000]:
                x.append((n, wC))
                Dataset = self.CreateDataset(wC, n)
                time0 = time.clock()
                self.ApproxBP(Dataset[0], Dataset[1], Dataset[2], 0.1)
                self.times.append(time.clock() - time0)
        plt.plot(range(1, 10), self.times, label=r"$AP$", color="green", linewidth=2)
        plt.xlabel(u"")
        plt.ylabel(u"time(second)")
        plt.title(u"Compare")
        plt.legend()
        plt.show()        

        
    def OneTime(self):
        w, v, C = self.CreateDataset(self.wCSV.get(), self.sizeSV.get())
        e = float(self.eSV.get())
        label = Label(self.Frame1, text='v:' + str(v))
        label.pack(side=TOP)
        label = Label(self.Frame1, text='w:' + str(w))
        label.pack(side=TOP)
        label = Label(self.Frame1, text='C:' + str(C))
        label.pack(side=TOP)
        label = Label(self.Frame1, text='e:' + str(e))
        label.pack(side=TOP)
        ds = self.DynamicProgramingBP(w, v, C)
        label = Label(self.Frame1, text='Dynamic:' + \
                      str(ds) + \
                      ",w=" + \
                      str(sum([w[i] * ds[i] for i in range(len(w))])) + \
                          ",v=" + \
                          str(sum([v[i] * ds[i] for i in range(len(v))])))
        label.pack(side=TOP)
        As = self.ApproxBP(w, v, C, e)
        label = Label(self.Frame1, text='Approx:' + \
                      str(As) + \
                      ",w=" + \
                      str(sum([w[i] * As[i] for i in range(len(w))])) + \
                          ",v=" + \
                          str(sum([v[i] * As[i] for i in range(len(v))])))
        label.pack(side=TOP)
        
    def DynamicProgramingBP(self, w, v, C):
        return DynamicProgramingBoolPacking(w, v, C)
    def ApproxBP(self, w, v, C, e):
        return ApproxBoolPacking(w, v, C, e)
    def CreateDataset(self, wC, size):
        return CreateDataset(wC, size)

        
def ApproxBoolPacking(w, v, C, e):
    K = len(w) / e
    vs = copy.deepcopy(v)
    vs.sort()
    vmax = vs[-1]
    return DynamicProgramingBoolPacking(w, [ceil(item * K / vmax) for item in v], C)


def DynamicProgramingBoolPacking(w, v, C):
    return BoolPackSolution(BoolPackCost(w, v, C), w, C)
def BoolPackCost(w, v, C):
    n = len(w)
    B = [[0 for i in range(C + 1)] for i in range(n)]
    for j in range(1, w[-1]):
        B[-1][j] = 0
    for j in range(w[-1], C + 1):
        B[-1][j] = v[-1]
    for i in range(n - 2, 0, -1):
        for j in range(w[i]):
            B[i][j] = B[i + 1][j]
        for j in range(w[i], C + 1):
            B[i][j] = B[i + 1][j - w[i]] + v[i]
            if B[i][j] < B[i + 1][j]:
                B[i][j] = B[i + 1][j]
    B[0][C] = B[1][C - w[0]] + v[0]
    if B[0][C] < B[1][C]:
        B[0][C] = B[1][C]
    return B
def BoolPackSolution(B, w, C):
    n = len(w)
    j = C
    x = [0 for i in range(n)]
    for i in range(n - 1):
        if B[i][j] == B[i + 1][j]:
            x[i] = 0
        else :
            x[i] = 1
            j = j - w[i]
    x[-1] = 0 if B[-1][j] == 0 else 1
    return x

            
def CreateDataset(wCMax, size):
    w = np.random.randint(1, wCMax, size)
    ws = w[:]
    ws.sort()
    C = np.random.randint(ws[-1], wCMax)
    v = np.random.randint(1, 100, size)
    return w, v, C

root = Tk()
app = App(root)
root.mainloop()