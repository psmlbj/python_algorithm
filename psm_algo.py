# -*- coding: cp936 -*-
import psmDS
import time


def my_min_max(*li):
    # just a test for "divide and conquer",has low efficiency
    if isinstance(li[0], list):
        li = li[0]
    def inter_min_max(*li):
        lilen = len(li)
        if lilen == 1:
            return li[0], li[0]
        elif lilen == 2:
            return (li[0], li[1]) if li[0] < li[1] else (li[1], li[0])
        else :
            k = lilen / 2
            min1, max1 = inter_min_max(*li[:k])
            min2, max2 = inter_min_max(*li[k:])
            return min1 if min1 < min2 else min2 , max1 if max1 > max2 else max2
    return inter_min_max(*li)
def min_max(*li):
    return min(*li), max(*li)
def max_factor(a, b):
    m, n = max(a, b), min(a, b)
    while n > 0:
        r = m % n
        m , n = n, r
    return m
def insert_sort(li):
    for i in range(1, len(li)):
        key = li[i]
        j = i - 1
        while j > -1 and li[j] > key:
            li[j + 1] = li[j]
            j = j - 1
        li[j + 1] = key
def int_product(x, y):
    pass
def change_one_dimension(x, only_advance=False):
    """
    x is a 0or1 list like [0,1,0,0],
    change it's one 0 to 1 to create it's children which like 
    [[1,1,0,0],
    [0,1,1,0],
    [0,1,0,1]].
    if only_advance is false ,only change 0 which behind the last 1.
    In this case ,the answer is 
    [[0,1,1,0],
    [0,1,0,1]].
    """
    ans = []
    start = 0
    if only_advance:
        for ix in xrange(-1, -len(x) - 1, -1):
            if x[ix] == 1:
                start = len(x) + ix
                break
    for i in range(start, len(x)):
        if x[i] == 0:
            y = x[:]
            y[i] = 1
            ans.append(y)
    return ans


class Search():
    @staticmethod
    def general_search(container, \
                       root, \
                       bounding_func, \
                       distance_to_object, \
                       operate_children, \
                       create_children, \
                       is_solution):
        # general search,the abstract of 4 search algorithm
        ds = container
        ds.add(root)
        for item in ds:
            print item
            if not is_solution(item):
                if bounding_func(item):
                    ds.adds(operate_children(create_children(item)))
            else:
                return item
        else:
            raise StopIteration
    @staticmethod
    def gbfs(root, bounding_func, create_children, is_solution):
        return Search.general_search(psmDS.PSMQueue(), \
                              root, \
                              bounding_func, \
                              lambda x:x, \
                              lambda x:x, \
                              create_children, \
                              is_solution)
    @staticmethod
    def gdfs(root, bounding_func, create_children, is_solution):
        return Search.general_search(psmDS.PSMStack(), \
                              root, \
                              bounding_func, \
                              lambda x:x, \
                              lambda x:x if (x.reverse()) is None else x, \
                              create_children, \
                              is_solution)
    @staticmethod
    def gclamb_hill(root, bounding_func, distance_to_object, \
                   create_children, is_solution):
        return Search.general_search(psmDS.PSMStack(), \
                              root, \
                              bounding_func, \
                              distance_to_object, \
                              lambda x:\
    x if (x.sort(key=distance_to_object, reverse=True)) is None else x, \
                              create_children, \
                              is_solution)
    @staticmethod
    def gbest_first(root, bounding_func, distance_to_object, \
                   create_children, is_solution):
        return Search.general_search(\
    psmDS.PSMHeap(lambda x, y:distance_to_object(x) < distance_to_object(y)), \
                              root, \
                              bounding_func, \
                              distance_to_object, \
                              lambda x:x, \
                              create_children, \
                              is_solution)
    @staticmethod
    def bfs(root, bounding_func, create_children, is_solution):
        # general broad first search
        DS = psmDS.Queue()
        DS.add(root)
        for item in DS:
            print item
            if is_solution(item):
                return item
            else:
                if bounding_func(item):
                    DS.adds(create_children(item))
        else:
            raise StopIteration
    @staticmethod
    def dfs(root, bounding_func, create_children, is_solution):
        # general depth first search
        DS = psmDS.PSMStack()
        DS.add(root)
        for item in DS:
            print item
            if is_solution(item):
                return item
            else:
                if bounding_func(item):
                    children = create_children(item)
                    children.reverse()
                    DS.adds(children)
        else:
            raise StopIteration

    @staticmethod
    def clamb_hill(root, \
                   bounding_func, \
                   distance_to_object, \
                   create_children, \
                   is_solution):
        DS = psmDS.PSMStack()
        DS.add(root)
        for item in DS:
            print item
            if is_solution(item):
                return item
            else:
                if bounding_func(item):
                    children = create_children(item)
                    children.sort(key=distance_to_object, reverse=True)
                    DS.adds(children)
        else:
            raise StopIteration
    @staticmethod
    def best_first(root, \
                   bounding_func, \
                   distance_to_object, \
                   create_children, \
                   is_solution):
        DS = psmDS.PSMHeap(lambda x, y:distance_to_object(x) < distance_to_object(y))
        DS.add(root)
        for item in DS:
            print item
            if is_solution(item):
                return item
            else:
                if bounding_func(item):
                    DS.adds(create_children(item))
        else:
            raise StopIteration



if __name__ == '__main__':

    S = [8, 5, 6, 2, 9, 6, 11, 22, 7, 15]
    root = [0 for i in range(len(S))]
    K = 43

    print "bfs\n", Search.gbfs(root, \
              lambda x:sum([x[i] * S[i] for i in range(len(S))]) < K, \
              lambda x:change_one_dimension(x, True), \
              lambda x:sum([x[i] * S[i] for i in range(len(S))]) == K)

    print "dfs\n", Search.gdfs(root, \
              lambda x:sum([x[i] * S[i] for i in range(len(S))]) < K, \
              lambda x:change_one_dimension(x, True), \
              lambda x:sum([x[i] * S[i] for i in range(len(S))]) == K)
    print "ch\n", Search.gclamb_hill(root, \
                     lambda x:sum([x[i] * S[i] for i in range(len(S))]) < K, \
                     lambda x:K - sum([x[i] * S[i] for i in range(len(S))]) , \
                     lambda x:change_one_dimension(x), \
                     lambda x:sum([x[i] * S[i] for i in range(len(S))]) == K)

    print "bf\n", Search.gbest_first(root, \
                     lambda x:sum([x[i] * S[i] for i in range(len(S))]) < K, \
                     lambda x:K - sum([x[i] * S[i] for i in range(len(S))]) , \
                     lambda x:change_one_dimension(x), \
                     lambda x:sum([x[i] * S[i] for i in range(len(S))]) == K)

