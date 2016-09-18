# -*- coding: cp936 -*-

from Tkinter import *
from cmath import polar
import time
import numpy as np
import matplotlib.pyplot as plt


class App:
    def __init__(self, master):
        # ��������
        self.Frame0 = Frame(master)
        self.Frame1 = Frame(master)
        self.Frame2 = Frame(master)
        self.Frame3 = Frame(master)
        self.Frame0.pack(side=TOP)
        self.Frame1.pack(side=LEFT)
        self.Frame2.pack(side=LEFT)
        self.Frame3.pack(side=LEFT)
        
        label = Label(self.Frame0, text=u'͹���㷨')
        label.pack(side=TOP)
        self.Frame01 = Frame(self.Frame0)
        self.Frame01.pack(side=LEFT)
        
        # ���ò���
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
        label = Label(self.Frame01, text=u'�������:')
        label.pack(side=LEFT)
        self.CompareSV = IntVar()
        self.CompareEntry = Entry(self.Frame01, textvariable=self.CompareSV)
        self.CompareSV.set('6')
        self.CompareEntry.pack(side=LEFT)
        self.CompareBt = Button(self.Frame01, text=u"�Ƚ������㷨", command=self.Compare)
        self.CompareBt.pack(side=LEFT)
        

        label = Label(self.Frame1, text=u'�����㷨')
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
        
        label = Label(self.Frame3, text=u'�����㷨')
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
    # �����������2*size���󣬵����������expand_ratio��
    randMat = randInt(min, max, d[0], d[1])
    # �Ѷ�ά����ת���ɵ�ļ���
    points = map(lambda x, y:(int(x), int(y)), *randMat)
    # ȥ�ظ�
    points = list(set(points))
    return points


def filterX(L, X):
    # return filter(lambda x:x!=X, L)
    return [i for i in L if i != X]


def randInt(min, max, *d):
    # ��������min��max֮��ľ���ά����dָ��
    return (np.random.rand(*d)) * (max - min) + min


def lineAndPoint(*p):
    # �ѵ���������뵽ǰ��������ɵ�ֱ����
    return (p[2][0] - p[0][0]) * (p[1][1] - p[0][1]) - (p[1][0] - p[0][0]) * (p[2][1] - p[0][1])

def is4thInTria(*p):
    '''���ĸ����Ƿ�������������ɵ������ε���'''
    if lineAndPoint(p[0], p[1], p[3]) * lineAndPoint(p[0], p[1], p[2]) >= 0 and\
    lineAndPoint(p[0], p[1], p[2]) != 0 and lineAndPoint(p[1], p[2], p[0]) != 0\
    and lineAndPoint(p[0], p[2], p[1]) != 0 and\
     lineAndPoint(p[1], p[2], p[3]) * lineAndPoint(p[1], p[2], p[0]) >= 0 and\
     lineAndPoint(p[0], p[2], p[3]) * lineAndPoint(p[0], p[2], p[1]) >= 0:
        return True
    return False;

def whichInCenter(*p):
    '''�����ĸ��㣬 ��һ��������������ɵ������ε���'''
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
    '''��points��ʱ������'''
    points.sort(key=lambda x:x[0])
    # �õ���������ҵĵ�
    left = points[0]
    right = points[-1]
    # �������˵㻭һ���ߣ������е�ֳ���������
    up = [i for i in points if lineAndPoint(left, right, i) >= 0]
    down = [i for i in points if lineAndPoint(left, right, i) < 0]
    # �ϰ��������к󣬺ϲ����°��
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
            # ���j��i�ظ�������j�Ѿ���ɾ������һ��j
            if pj in [p0, pi, False]:
                continue
            for pk in points:
                if pk in [p0, pi, pj, False]:
                    continue
                # �����һ����������������ɵ������ε��У���ans�����Ǹ��㣬������none
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
            # ���j��i�ظ�������j�Ѿ���ɾ������һ��j
            if pj in [pi, False]:
                continue
            for pk in points:
                if pk in [pi, pj, False]:
                    continue
                for pl in points:
                    if pl in [pi, pj, pk, False]:
                        continue
                    # �����һ����������������ɵ������ε��У���ans�����Ǹ��㣬������none
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
    # ��pΪԭ�㣬paΪx�ᣬ�����µĻ�������ɾ���Ϊ
    return orthonormalize([[a[0] - p[0], p[1] - a[1]], [a[1] - p[1], a[0] - p[0]]])
def newCoordinateInNewBasis(p, TransitionI, x):
    # p��������ϵ��ԭ����������ϵ�µ�����
    # TransitionI�ǹ��Ⱦ����׼��֮������
    # x�ǵ��ھ�����ϵ������
    # ����x��������ϵ������
    p = np.array([[p[0]], [p[1]]])
    x = np.array([[x[0]], [x[1]]])
    xn = TransitionI * (x - p)
    xn = xn.getA()
    return xn[0][0], xn[1][0]

def DivideAndConquerCH(app, points):
    if len(points) <= 3:
        return easternSort(points)
    points.sort(key=lambda x:x[0])
    # ����
    PL = points[:len(points) / 2]
    PR = points[len(points) / 2:]
    # ���
    QL = DivideAndConquerCH(app, PL)
    QR = DivideAndConquerCH(app, PR)
    # �ϲ�
    # �ҵ���������С�ĵ�a������һ���ڵ�p
    QL.sort(key=lambda x:x[1])
    a = QL[0]
    b = QL[1]
    p = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0 + 1)
    # ��pΪԭ�㣬paΪx�ᣬ�����µ�����ϵ���õ����Ⱦ���������
    TransitionI = np.mat(transitionMatrix(p, a)).I
    # ��ÿ������������ϵ�µ����꣬��һ������pΪ���㣬paΪ����ļ�����
    QLP = [(i[0], i[1], Polar((0, 0), newCoordinateInNewBasis(p, TransitionI, i))) for i in QL]
    QRP = [(i[0], i[1], Polar((0, 0), newCoordinateInNewBasis(p, TransitionI, i))) for i in QR]
    # �ϲ�֮�󰴼�������
    QLP.extend(QRP)
    QLP.sort(key=lambda x:x[2][1])
    # ����GrahamScanCH����
    return GrahamScanCH(QLP)

    

def Polar(p0, p):
    # ��p0Ϊ���㣬��x����Ϊ���ᣬ��p�ļ�����
    return polar(p[0] - p0[0] + (p[1] - p0[1]) * 1j)

def GSPreprocess(points):
    # ������������
    points.sort(key=lambda x:x[1])
    # �õ���������С�ĵ㣬��Ϊ����
    p0 = points[0]
    # ��չ�б�����ÿ����ļ����꣬������p0
    points = map(lambda x:(x[0], x[1], Polar(p0, x)), points[1:])
    # ������������
    points.sort(key=lambda x:x[2][1])
    # ������ͬ�ĵ�ֻ����һ��ֱ������
    # ����ǰ�����һ������Ԫ��
    points.insert(0, (0, 0, (1, -1)))
    j = 0
    for i in range(1, len(points)):
        # ���ǲ�ͬ�������¼���Ѱ��
        if points[j][2][1] != points[i][2][1]:
            j = i
        else:
            # ������ͬ����ɾ��ֱ��С��
            if points[j][2][0] > points[i][2][0]:
                points[i] = False
            else:
                points[j] = False
                j = i
    # ɾ������Ԫ��
    del(points[0])
    # p0�ļ��������ֹ�ָ����
    points.insert(0, (p0[0], p0[1], (0, 0)))
    return filterX(points, False)

def GrahamScanCH(points):
    # Ԥ��������
    P = GSPreprocess(points)
    if len(P) <= 1:
        print "CH is void"
    Q = P[:3]
    for i in range(3, len(P)):
        # �ж�P[0]��Q[-2]�Ƿ���P[i]Q[-1]����߶ε�ͬһ�� �������ջ
        while lineAndPoint(P[i], Q[-1], P[0]) * lineAndPoint(P[i], Q[-1], Q[-2]) < 0:
            Q.pop()
        Q.append(P[i])
    return map(lambda x:(x[:2]), Q)


root = Tk()
app = App(root)
root.mainloop()
