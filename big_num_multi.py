# -*- coding: cp936 -*-
#输入两个n位二进制整数，输出其乘积
from __builtin__ import str
import merge_sort
def bin2dec(bi):
    ans=0
    for i in str(bi):
        ans *= 2
        ans += int(i)
    return ans

def int_multi(int1,int2):
    int1len = len(int1)
    int2len = len(int2)

    #把int1分成a和b两半，把int2分成c和d两半
    a=int1[:int1len/2]
    b=int1[int1len/2:]
    c=int2[:int2len/2]
    d=int2[int2len/2:]

    ac = bin2dec(a)*bin2dec(c)
    bd = bin2dec(b)*bin2dec(d)
    a_bc_d = (bin2dec(a)-bin2dec(b))*(bin2dec(c)-bin2dec(d))

    return (ac<<int1len) + ( (ac+bd-a_bc_d) << (int1len/2) ) + bd
    #answer1
    #answer1 = bin2dec(int1)*bin2dec(int2)

print int_multi('1011','1001')
li=[1,1,4,2,7,5,9,0,8,8]
print merge_sort.merge_sort(li),li