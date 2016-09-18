# -*- coding: cp936 -*-
import os
import shutil
import string
def LCS(X, Y):
    if not (isinstance(X, basestring) and isinstance(Y, basestring)):
        raise TypeError("is not basestring.")
    C = [[[0, ' '] for i in range(len(Y) + 1)] for j in range(len(X) + 1)]
    for i in C:
        print i
    for jx in range(len(X)):
        print jx
        for iy in range(len(Y)):
            print jx,iy,C[jx][iy][0]
            if Y[iy] == X[jx]:
                C[jx + 1][iy + 1] = [C[jx][iy][0] + 1, "#"]
            else:
                if C[jx + 1][iy][0] > C[jx][iy + 1][0]:
                    C[jx + 1][iy + 1] = [C[jx + 1][iy][0] , "<"]
                else:
                    C[jx + 1][iy + 1] = [C[jx][iy + 1][0] , "^"]
    for i in C:
        print i

X = "ABCBDAB"
Y = "BDCABA"
LCS(X, Y)
