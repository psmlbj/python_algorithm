# -*- coding: cp936 -*-

from Tkinter import *
from cmath import polar
import time
import numpy as np
import matplotlib.pyplot as plt


class App:
    def __init__(self, master):
        # 划分区域
        self.Frame0 = Frame(master)
        self.Frame1 = Frame(master)
        self.Frame2 = Frame(master)
        self.Frame3 = Frame(master)
        self.Frame0.pack(side=TOP)
        self.Frame1.pack(side=LEFT)
        self.Frame2.pack(side=LEFT)
        self.Frame3.pack(side=LEFT)
        
        label = Label(self.Frame0, text=u'凸包算法')
        label.pack(side=TOP)
        self.Frame01 = Frame(self.Frame0)
        self.Frame01.pack(side=LEFT)
        
        # 设置参数
        label = Label(self.Frame01, text=u'min:')
        label.pack(side=LEFT)
        self.minSV = IntVar()
        self.minEntry = Entry(self.Frame01, textvariable=self.minSV, state='readonly')
        self.minSV.set('10')
        self.minEntry.pack(side=LEFT)
        
        label = Label(self.Frame01, text=u'max:')
        label.pack(side=LEFT)
        self.maxSV = IntVar()
        self.maxEntry = Entry(self.Frame01, textvariable=self.maxSV, state='readonly')
        self.maxSV.set('400')
        self.maxEntry.pack(side=LEFT)
        
        self.Canvas_size = int(self.minSV.get()) + int(self.maxSV.get())
        
        label = Label(self.Frame01, text=u'size:')
        label.pack(side=LEFT)
        self.sizeSV = IntVar()
        self.sizeEntry = Entry(self.Frame01, textvariable=self.sizeSV)
        self.sizeSV.set('1000')
        self.sizeEntry.pack(side=LEFT)
        
        label = Label(self.Frame01, text=u'               ')
        label.pack(side=LEFT)
        label = Label(self.Frame01, text=u'试验次数:')
        label.pack(side=LEFT)
        self.CompareSV = IntVar()
        self.CompareEntry = Entry(self.Frame01, textvariable=self.CompareSV)
        self.CompareSV.set('6')
        self.CompareEntry.pack(side=LEFT)
        self.CompareBt = Button(self.Frame01, text=u"比较三种算法", command=self.Compare)
        self.CompareBt.pack(side=LEFT)
        

        label = Label(self.Frame1, text=u'蛮力算法')
        label.pack(side=TOP)
        self.BFCanvas = Canvas(self.Frame1, bg='white', height=str(self.Canvas_size), width=str(self.Canvas_size))
        self.BFFrame = Frame(self.Frame1)
        self.BFFrame.pack(side=BOTTOM)
        self.BFCreateDataBt = Button(self.BFFrame, text="CreateData", command=self.BFCreateDataset)
        self.BFCreateDataBt.pack(side=TOP)
        self.BFRunBt = Button(self.BFFrame, text="run", command=self.bruteForceCH)
        self.BFRunBt.pack(side=TOP)
        self.BFCanvas.pack(side=LEFT)
        
        label = Label(self.Frame2, text=u'GrahamScan')
        label.pack(side=TOP)
        self.GSCanvas = Canvas(self.Frame2, bg='white', height=str(self.Canvas_size), width=str(self.Canvas_size))
        self.GSFrame = Frame(self.Frame2)
        self.GSFrame.pack(side=BOTTOM)
        self.GSCreateDataBt = Button(self.GSFrame, text="CreateData", command=self.GSCreateDataset)
        self.GSCreateDataBt.pack(side=TOP)
        self.GSRunBt = Button(self.GSFrame, text="run", command=self.GrahamScanCH)
        self.GSRunBt.pack(side=TOP)
        self.GSCanvas.pack(side=LEFT)
        
        label = Label(self.Frame3, text=u'分治算法')
        label.pack(side=TOP)
        self.DCCanvas = Canvas(self.Frame3, bg='white', height=str(self.Canvas_size), width=str(self.Canvas_size))
        self.DCFrame = Frame(self.Frame3)
        self.DCFrame.pack(side=BOTTOM)
        self.DCCreateDataBt = Button(self.DCFrame, text="CreateData", command=self.DCCreateDataset)
        self.DCCreateDataBt.pack(side=TOP)
        self.DCRunBt = Button(self.DCFrame, text="run", command=self.DivideAndConquerCH)
        self.DCRunBt.pack(side=TOP)
        self.DCCanvas.pack(side=LEFT)
        

    def Compare(self):
        n = int(self.CompareSV.get()) + 1
        
        self.times = []
        for i in range(1, n):
            Dataset = CreateDataset(self.minSV.get(), self.maxSV.get(), 2, i * 1000)
            time0 = time.clock()
            DivideAndConquerCH(self, Dataset)
            self.times.append(time.clock() - time0)
        plt.figure(figsize=(10, 7))
        plt.plot(range(1, n), self.times, label=r"$DC$", color="red", linewidth=2)

        self.times = []
        for i in range(1, n):
            Dataset = CreateDataset(self.minSV.get(), self.maxSV.get(), 2, i * 1000)
            time0 = time.clock()
            GrahamScanCH(Dataset)
            self.times.append(time.clock() - time0)
        plt.plot(range(1, n), self.times, label=r"$GS$", color="green", linewidth=2)
        
        self.times = []
        for i in range(1, n):
            Dataset = CreateDataset(self.minSV.get(), self.maxSV.get(), 2, i * 50)
            time0 = time.clock()
            bruteForceCH(Dataset)
            self.times.append(time.clock() - time0)
        plt.plot(range(1, n), self.times, label=r"$BF$", color="blue", linewidth=2)
        plt.xlabel(u"Dots(K for DC,GS\n50 for BF)")
        plt.ylabel(u"time(second)")
        plt.title(u"Compare")
        plt.legend()
        plt.show()
        

    def DivideAndConquerCH(self):
        displayCH(self.DCCanvas, DivideAndConquerCH(self, self.points))
    def GrahamScanCH(self):
        displayCH(self.GSCanvas, GrahamScanCH(self.points))
    def bruteForceCH1(self):
        displayCH(self.BFCanvas, bruteForceCH1(self.points))
    def bruteForceCH(self):
        displayCH(self.BFCanvas, bruteForceCH(self.points))            

    def BFCreateDataset(self):
        self.points = CreateDataset(self.minSV.get(), self.maxSV.get(), 2, self.sizeSV.get())
        displayPoints(self.BFCanvas, self.points)
    def GSCreateDataset(self):
        self.points = CreateDataset(self.minSV.get(), self.maxSV.get(), 2, self.sizeSV.get())
        displayPoints(self.GSCanvas, self.points)
    def DCCreateDataset(self):
        self.points = CreateDataset(self.minSV.get(), self.maxSV.get(), 2, self.sizeSV.get())
        displayPoints(self.DCCanvas, self.points)
        

            

    
    
def CreateDataset(min, max, *d):
    # 产生随机整型2*size矩阵，点的坐标扩大expand_ratio倍
    randMat = randInt(min, max, d[0], d[1])
    # 把二维矩阵转换成点的集合
    points = map(lambda x, y:(int(x), int(y)), *randMat)
    # 去重复
    points = list(set(points))
    return points


def filterX(L, X):
    # return filter(lambda x:x!=X, L)
    return [i for i in L if i != X]


def randInt(min, max, *d):
    # 产生介于min与max之间的矩阵，维度有d指定
    return (np.random.rand(*d)) * (max - min) + min


def lineAndPoint(*p):
    # 把第三个点带入到前两个点组成的直线中
    return (p[2][0] - p[0][0]) * (p[1][1] - p[0][1]) - (p[1][0] - p[0][0]) * (p[2][1] - p[0][1])

def is4thInTria(*p):
    '''第四个点是否在另外三个组成的三角形当中'''
    if lineAndPoint(p[0], p[1], p[3]) * lineAndPoint(p[0], p[1], p[2]) >= 0 and\
    lineAndPoint(p[0], p[1], p[2]) != 0 and lineAndPoint(p[1], p[2], p[0]) != 0\
    and lineAndPoint(p[0], p[2], p[1]) != 0 and\
     lineAndPoint(p[1], p[2], p[3]) * lineAndPoint(p[1], p[2], p[0]) >= 0 and\
     lineAndPoint(p[0], p[2], p[3]) * lineAndPoint(p[0], p[2], p[1]) >= 0:
        return True
    return False;

def whichInCenter(*p):
    '''任意四个点， 哪一个在另外三个组成的三角形当中'''
    if is4thInTria(p[0], p[3], p[2], p[1]):
        return p[1]
    if is4thInTria(p[0], p[1], p[3], p[2]):
        return p[2]
    if is4thInTria(p[0], p[1], p[2], p[3]):
        return p[3]
    if is4thInTria(p[3], p[1], p[2], p[0]):
        return p[0]
    return None


def easternSort(points):
    '''对points逆时针排序'''
    points.sort(key=lambda x:x[0])
    # 得到最左和最右的点
    left = points[0]
    right = points[-1]
    # 以两个端点画一条线，把所有点分成上下两半
    up = [i for i in points if lineAndPoint(left, right, i) >= 0]
    down = [i for i in points if lineAndPoint(left, right, i) < 0]
    # 上半逆序排列后，合并到下半后
    up.reverse()
    down.extend(up)
    return down

def bruteForceCH(points):
    points.sort(key=lambda x:x[1])
    p0 = points[0]
    for pi in points:
        if pi in [p0, False]:
            continue
        for pj in points:
            # 如果j与i重复，或者j已经被删除，则换一个j
            if pj in [p0, pi, False]:
                continue
            for pk in points:
                if pk in [p0, pi, pj, False]:
                    continue
                # 如果有一个点在另三个点组成的三角形当中，则ans就是那个点，否则是none
                ans = whichInCenter(pi, pj, pk, p0)
                # print "4 ", ans
                if ans is None:
                    pass
                else:
                    try:
                        points[points.index(ans)] = False
                    except:
                        pass
    return easternSort(filterX(points, False))

def bruteForceCH1(points):
    for pi in points:
        if pi in [False]:
            continue
        for pj in points:
            # 如果j与i重复，或者j已经被删除，则换一个j
            if pj in [pi, False]:
                continue
            for pk in points:
                if pk in [pi, pj, False]:
                    continue
                for pl in points:
                    if pl in [pi, pj, pk, False]:
                        continue
                    # 如果有一个点在另三个点组成的三角形当中，则ans就是那个点，否则是none
                    ans = whichInCenter(pi, pj, pk, pl)
                    # print "4 ", ans
                    if ans is None:
                        pass
                    else:
                        try:
                            points[points.index(ans)] = False
                        except:
                            pass
    return easternSort(filterX(points, False))


def displayCH(canvas, points):
    for i in range(len(points)):
        canvas.create_line(points[i - 1], points[i])

def displayPoints(canvas, points):
    map(lambda x:canvas.create_oval(x[0] - 1, x[1] - 1, x[0] + 1, x[1] + 1), points)
   
   
def SquareSum(a):
    return sum([i * i for i in a])
def orthonormalize(A):
    from cmath import sqrt
    return [[(float(i) / sqrt(SquareSum(x))).real for i in x] for x in A]
def transitionMatrix(p, a):
    # 以p为原点，pa为x轴，建立新的基，则过渡矩阵为
    return orthonormalize([[a[0] - p[0], p[1] - a[1]], [a[1] - p[1], a[0] - p[0]]])
def newCoordinateInNewBasis(p, TransitionI, x):
    # p是新坐标系的原点在老坐标系下的坐标
    # TransitionI是过度矩阵标准化之后求逆
    # x是点在旧坐标系的坐标
    # 返回x在新坐标系的坐标
    p = np.array([[p[0]], [p[1]]])
    x = np.array([[x[0]], [x[1]]])
    xn = TransitionI * (x - p)
    xn = xn.getA()
    return xn[0][0], xn[1][0]

def DivideAndConquerCH(app, points):
    if len(points) <= 3:
        return easternSort(points)
    points.sort(key=lambda x:x[0])
    # 划分
    PL = points[:len(points) / 2]
    PR = points[len(points) / 2:]
    # 求解
    QL = DivideAndConquerCH(app, PL)
    QR = DivideAndConquerCH(app, PR)
    # 合并
    # 找到纵坐标最小的点a，再找一个内点p
    QL.sort(key=lambda x:x[1])
    a = QL[0]
    b = QL[1]
    p = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0 + 1)
    # 以p为原点，pa为x轴，建立新的坐标系，得到过度矩阵的逆矩阵
    TransitionI = np.mat(transitionMatrix(p, a)).I
    # 求每个点在新坐标系下的坐标，进一步求以p为极点，pa为极轴的极坐标
    QLP = [(i[0], i[1], Polar((0, 0), newCoordinateInNewBasis(p, TransitionI, i))) for i in QL]
    QRP = [(i[0], i[1], Polar((0, 0), newCoordinateInNewBasis(p, TransitionI, i))) for i in QR]
    # 合并之后按极角排序
    QLP.extend(QRP)
    QLP.sort(key=lambda x:x[2][1])
    # 调用GrahamScanCH处理
    return GrahamScanCH(QLP)

    

def Polar(p0, p):
    # 以p0为极点，以x方向为极轴，求p的极坐标
    return polar(p[0] - p0[0] + (p[1] - p0[1]) * 1j)

def GSPreprocess(points):
    # 按纵坐标排序
    points.sort(key=lambda x:x[1])
    # 得到纵坐标最小的点，作为极点
    p0 = points[0]
    # 扩展列表，加入每个点的极坐标，不包括p0
    points = map(lambda x:(x[0], x[1], Polar(p0, x)), points[1:])
    # 按极坐标排序
    points.sort(key=lambda x:x[2][1])
    # 极角相同的点只保留一个直径最大的
    # 在最前面插入一个辅助元素
    points.insert(0, (0, 0, (1, -1)))
    j = 0
    for i in range(1, len(points)):
        # 极角不同，则往下继续寻找
        if points[j][2][1] != points[i][2][1]:
            j = i
        else:
            # 极角相同，则删除直径小的
            if points[j][2][0] > points[i][2][0]:
                points[i] = False
            else:
                points[j] = False
                j = i
    # 删除辅助元素
    del(points[0])
    # p0的极坐标是手工指定的
    points.insert(0, (p0[0], p0[1], (0, 0)))
    return filterX(points, False)

def GrahamScanCH(points):
    # 预处理数据
    P = GSPreprocess(points)
    if len(P) <= 1:
        print "CH is void"
    Q = P[:3]
    for i in range(3, len(P)):
        # 判断P[0]和Q[-2]是否在P[i]Q[-1]组成线段的同一侧 ，否则出栈
        while lineAndPoint(P[i], Q[-1], P[0]) * lineAndPoint(P[i], Q[-1], Q[-2]) < 0:
            Q.pop()
        Q.append(P[i])
    return map(lambda x:(x[:2]), Q)


root = Tk()
app = App(root)
root.mainloop()
