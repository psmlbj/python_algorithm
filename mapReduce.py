def mergelist(*ls):
    l = []
    for i in ls:
        l.extend(i)
    return l
def grouping(data):
    di = {}
    for i,ix in data:
        if di.get(i) is None:
            di[i] = [ix]
        else:
            di[i].append(ix)
    return di
def mapReduce(mapf,reducef,data):
    mid = mapf(data)
    di = grouping(mid)
    mid2 = [[reducef(k,v)] for k,v in di.items()]
    return mergelist(*mid2)

if __name__ == '__main__':
    # word 
    data = 'are are boy buy cat boy are'
    mymap = lambda data:[[item,1] for item in data.split()]
    myreduce = lambda k,vs:[k,sum(vs)]
    print mapReduce(mymap,myreduce,data)
    

