import sys

def biSearch(l, x):
    low,high = 0,len(l) - 1
    while low <= high:
        mid = (low + high) / 2
        midval = l[mid]

        if midval < x:
            low = mid + 1
        elif midval > x:
            high = mid - 1
        else:
            return mid
    return None
def biSearchRec(l, x, low, high):
    if low > high:
        return None
    mid = (low + high) / 2
    if l[mid] == x:
        return mid
    return (biSearchRec(l, x, mid + 1, high) if l[mid] < x
            else biSearchRec(l, x, low, mid - 1))


if __name__ == "__main__":
    l = range(10)
    x = 9
    print biSearch(l, x), biSearchRec(l, x, 0, len(l) - 1)

