# -*- coding: cp936 -*-
from __builtin__ import enumerate
from _random import Random
import os
import random

import psm_algo
import psmDS
import math


def merge_sort(li):
    if len(li) == 1:
        # print 11,li
        return li
    elif len(li) == 2:
        # print 21,li
        if li[0] < li[1]:
            # print 22,[li[0],li[1]]
            return [li[0], li[1]]
        else :
            # print 23,[li[1],li[0]]
            return [li[1], li[0]]
    else :
        # print 31,li
        k = len(li) / 2
        # print '3k',k
        li1 = merge_sort(li[:k])
        li2 = merge_sort(li[k:])
        # print 32,li1,li2

        # 合并
        li3 = []
        pointer1 = 0
        pointer2 = 0
        li1len = len(li1)
        li2len = len(li2)
        
        # 使用两个指针依次读取两个列表
        while(pointer1 < li1len and pointer2 < li2len):
            if li1[pointer1] < li2[pointer2]:
                li3.append(li1[pointer1])
                pointer1 += 1
            else:
                li3.append(li2[pointer2])
                pointer2 += 1
        if pointer1 < li1len:
            li3 += li1[pointer1:]
        else:
            li3 += li2[pointer2:]
        return li3


if __name__ == '__main__':
    li = [1, 1, 4, 2, 7, 5, 9, 0, 8, 8]
    print li
    print merge_sort(li)
    print li, "0"*3
    print filter(lambda x:x == 1, li)
    print os.path.splitext("dfg.gfhfg.")
    print "dfdf-jjn".split("-")
    h = psmDS.Heap(lambda x, y:x < y)
    h.adds([25, 20, 22, 17, 19, 10, 12, 15, 7, 9, 18])
    print h.heap
    for i in h:
        print i
    print [23, 4].reverse(), r"http://{%s,%s}" % (3, 4) + "f", [5, 1, 2][0:1]
    x = [[]]
    print x[0]
    x[0].append([4, 6])
    x[0].append([4, 6])
    print random.randint(-400,400)
    num={'.' : 0,'x' :0,'o' : 0}
    num['.']=5
    print dict([['F','F--F+F'],['X','FX-FY']])
    print math.pi,sum([2,3,4])
