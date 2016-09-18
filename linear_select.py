# -*- coding: cp936 -*-
"""
线性时间选择算法
"""

def linear_select(li, k):
    if len(li) < 20:
        li.sort()
        return li[k - 1]
    S = []
    for i, e in enumerate(li):
        if i % 5 == 0:
            S.append([])
        S[i / 5].append(e)
    for item in S:
        item.sort()
    print S

print linear_select(range(30, 1, -1), 3)
    
