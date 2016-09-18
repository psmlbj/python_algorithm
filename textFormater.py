import sys

def howmanywords(words, maxL):
    lens = map(lambda x:len(x), words)
    countW = lens[0]
    countN = 1
    for item in lens[1:]:
        if (countW + item + 1) <= maxL:
            countW += (item + 1)
            countN += 1
            continue
        else:
            break
    return countW, countN
def padspaces(words, maxL):
    lens = map(lambda x:len(x), words)
    spaceN = maxL - sum(lens) - len(words) + 1
    if spaceN == 0:
        return ' '.join(words)
    if len(words) == 1:
        return words[0] + ' ' * spaceN
    seq = [spaceN / (len(words) - 1) for i in range(len(words) - 1)]
    for i in range(spaceN - sum(seq)):
        seq[i] += 1
    seq.append(0)
    return ' '.join([words[i] + ' ' * seq[i] for i in range(len(words))])
def formater(setance, maxL):
    # for a setance,we make it looks better.
    words = setance.split()
    lines = []
    while len(words) > 0:
        countW, countN = howmanywords(words, maxL)
        lines.append(words[0:countN])
        words = words[countN:]
    for line in lines[0:-1]:
        print padspaces(line, maxL)
    return ' '.join(lines[-1])


if __name__ == '__main__':
    setance = '''when that day i hear your voice, 
    i have some special feeling. if day in the future, 
    this love will be comeing true.'''
    maxL = 20
    print formater(setance, maxL)

