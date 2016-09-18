import math
import time
def isPrime(x):
    for i in range(2,int(math.sqrt(x))+1):
        if x % i == 0:
            return False
    return False if x in [0,1] else True
print time.time()
print isPrime(0)
print time.time()