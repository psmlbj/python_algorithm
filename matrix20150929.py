# -*- coding: cp936 -*-
# ��������nλ�����������������˻�
import copy
import time
'''��������ģ��

ע������һ���Ƕ�ά�ģ�����ĺ����������б�
�����������Ŷ���0��ʼ
'''

def getMin(*li):
    lilen = len(li)
    if lilen == 1:
        return li[0]
    elif lilen == 2:
        return li[0] if li[0] < li[1] else li[1]
    else :
        k = lilen / 2
        min1 = getMin(*li[:k])
        min2 = getMin(*li[k:])
        return min1 if min1 < min2 else min2

def getMax(*li):
    lilen = len(li)
    if lilen == 1:
        return li[0], li[0]
    elif lilen == 2:
        return li[0] if li[0] > li[1] else li[1]
    else :
        k = lilen / 2
        max1 = getMax(*li[:k])
        max2 = getMax(*li[k:])
        return max1 if max1 > max2 else max2
    
def getMinMax(*li):
    lilen = len(li)
    if lilen == 1:
        return li[0], li[0]
    elif lilen == 2:
        return (li[0], li[1]) if li[0] < li[1] else (li[1], li[0])
    else :
        k = lilen / 2
        min1, max1 = getMinMax(*li[:k])
        min2, max2 = getMinMax(*li[k:])
        return min1 if min1 < min2 else min2 , max1 if max1 > max2 else max2

def isVector(li):
    u"""���ֲ�������������������"""
    try:
        len(li)
    except TypeError:
        return False
    n = 0
    for i in range(len(li)):
        try:
            len(li[i])
        except TypeError:
            n += 1
    if len(li) == n:
        return True
    else:
        return False
def inner_product(a, b):
    if isVector(a) and isVector(b):
        if len(a) != len(b):
            raise Exception("����������ģ��ͬ")
        return sum(map(lambda i, j:i * j, a, b))
    else:
        raise Exception("������������")




class MatrixPSM:
    def __init__(self, m):
        self.rawMatrix = m

    def rowNum(self):
        return len(self.rawMatrix)
    def colNum(self):
        return len(self.rawMatrix[0])
       
    @staticmethod
    def isRawMatrix(li):
        u'''�ж�list�Ƿ��Ǿ���'''
        if isVector(li):
            return False
        f = map(lambda x:len(x), li)
        if f.count(len(li[0])) == len(li):
            return True
        else:
            return False
            
    def transpose(self):
        if self.rawMatrix == [[]]:
            return [[]]
        # �б�����ٶȸ���һ��
        # return map(lambda *x:list(x), *self.rawMatrix)
        return MatrixPSM([[row[i] for row in self.rawMatrix] for i in range(len(self.rawMatrix[0]))])

    @staticmethod
    def isRowSizeEqual(*matrix):
        if map(lambda item:item.rowNum(), matrix).count(matrix[0].rowNum()) != len(matrix):
            return False
        else:
            return True
    @staticmethod
    def isColumnSizeEqual(*matrix):
        if map(lambda item:item.colNum(), matrix).count(matrix[0].colNum()) != len(matrix):
            return False
        else:
            return True
    @staticmethod
    def isSizeEqual(*matrix):
        if False in [isinstance(item, MatrixPSM) for item in matrix]:
            raise Exception("�������Ǿ���")
        elif not (MatrixPSM.isRowSizeEqual(*matrix) and MatrixPSM.isColumnSizeEqual(*matrix)):
            return False
        else:
            return True
    
    def isSquareMatrix(self):
        if self.rowNum() == self.colNum():
            return True
        else:
            return False
    
    @staticmethod
    def list2matrix(*tupl):
        u'''��һ�������б�ת��Ϊ����
        ��������������������������������������������������������������������������������������������������������������
        '''
        length = map(lambda x:len(x), tupl)
        if length.count(len(tupl[0])) == len(tupl):
            return MatrixPSM(list(tupl))
        else :
            raise Exception("������ģ��ͬ")
        
    def printMatrix(self):
        print "["
        for i in self.rawMatrix:
            print " ", i
        print "]"
    
    def getSubMatrix(self, row, column):
        # matrix = copy.deepcopy(matrix)
        u'''����row�е��У�column�е�����ɵ��Ӿ���
            
        '''
        if row == [] or column == []:
            return [[]]
        try:
            return MatrixPSM([[self.rawMatrix[r][c] for c in column] for r in row])
            # return transpose(map(lambda y:transpose(map(lambda x:matrix[x], row))[y], column))
        except IndexError:
            raise Exception("��������")
    
    @staticmethod
    def joinMatrix(m1, m2, direction):
        u'''������������
        direction��0����������ӣ���1����������'''
        if False in [isinstance(item, MatrixPSM) for item in [m1, m2]]:
            raise Exception("�������Ǿ���")
        if direction == 0:
            if m1.rowNum() != m2.rowNum():
                raise Exception("���������ģ��ͬ�����ܺ�������")
            else:
                return MatrixPSM(map(lambda x, y:x + y, m1.rawMatrix, m2.rawMatrix))
        elif direction == 1:
            if m1.colNum() != m2.colNum():
                raise Exception("���������ģ��ͬ��������������")
            else:
                return MatrixPSM(m1.rawMatrix + m2.rawMatrix)
        else:
            raise Exception("����direction����ֻ��Ϊ0��1")
    
    @staticmethod
    def zeros(m, n):
        u'''���ɾ���Ԫ��ȫΪ0'''
        return MatrixPSM([[0 for i in range(n)] for i in range(m)])
    @staticmethod
    def eyes(m, n):
        u'''���ɵ�λ����'''
        a = MatrixPSM.zeros(m, n)
        for i in range(getMin(m, n)):
            a.rawMatrix[i][i] = 1
        return a
    @staticmethod
    def ones(m, n):
        u'''���ɾ���Ԫ��ȫΪ1'''
        return MatrixPSM([[1 for i in range(n)] for i in range(m)])
    

    
    def diag(self):
        # matrix = copy.deepcopy(matrix)
        u'''��ȡ�Խ����ϵ�Ԫ�أ����һ���о���
        
        '''
        if self.rawMatrix == [[]]:
            return [[]]
        ans = []
        j = 0
        for i in self.rawMatrix:
            try:
                ans.append(i[j])
                j += 1
            except IndexError:
                break
        return MatrixPSM([ans])
    
    def diagonalize(self):
        u'''�����б任�Խǻ�'''
        a = copy.deepcopy(self.rawMatrix)
        for j in range(getMin(len(a), len(a[0]))):
            if a[j][j] == 0:
                for i in range(j + 1, len(a)):
                    if a[i][j] != 0:
                        a[j], a[i] = a[i], a[j]
                        # ���к󣬰�һ�мӸ���
                        for i in range(len(a[j])):
                            a[1][i] = -a[1][i]         
                        break
            if a[j][j] == 0:
                return a
            for i in range(j):
                a[i] = list(map(lambda x, y:x - y * a[i][j] / float(a[j][j]), a[i], a[j]))
            for i in range(j + 1, len(a)):
                a[i] = list(map(lambda x, y:x - y * a[i][j] / float(a[j][j]), a[i], a[j]))
        return MatrixPSM(a)
    
    def normalize(self):
        # matrix = copy.deepcopy(self.rawMatrix)
        u'''�����б任����׼��'''
        diag = self.diagonalize()
        for i in range(getMin(self.rowNum(), self.colNum())):
            if diag.rawMatrix[i][i] != 0:
                diag.rawMatrix[i] = [item / float(diag.rawMatrix[i][i]) for item in diag.rawMatrix[i]]
        return diag
    
    def rank(self):
        # matrix = copy.deepcopy(matrix)
        u'''��������'''
        diag = self.diagonalize()
        rank = 0
        for i in range(len(self.rawMatrix)):
            if inner_product([1] * self.colNum(), diag.rawMatrix[i]) != 0:
                rank += 1
        return rank
    
    def determinant(self):
        # matrix = copy.deepcopy(matrix)
        u'''������ʽֵ'''
        if not self.isSquareMatrix():
            raise Exception("�������Ƿ���")
        diag = self.diagonalize()
        b = 1
        for i in range(diag.rowNum()):
            b *= diag.rawMatrix[i][i]
        return b
    
    @staticmethod
    def plus(*matrix):
        u'''�������'''
        if False in [isinstance(item, MatrixPSM) for item in matrix]:
            raise Exception("�������Ǿ���")
        if not MatrixPSM.isSizeEqual(*matrix):
            raise Exception("������ģ��ͬ")
        return MatrixPSM(map(lambda *row:map(lambda *column:sum(column), *row), *[mat.rawMatrix for mat in matrix]))
    @staticmethod
    def minus(a, b):
        u'''�������'''
        if False in [isinstance(item, MatrixPSM) for item in [a, b]]:
            raise Exception("�������Ǿ���")
        if not MatrixPSM.isSizeEqual(a, b):
            raise Exception("������ģ��ͬ")
        return MatrixPSM(map(lambda x, y:map(lambda i, j:i - j, x, y), a.rawMatrix, b.rawMatrix))
    
    @staticmethod
    def multiply2(a, b):
        def multiplySameEvenSquare(a, b):
            n = a.rowNum()
            r1k = range(n / 2)
            rkn = range(n / 2 , n)
            A11 = a.getSubMatrix(r1k, r1k)
            A12 = a.getSubMatrix(r1k, rkn)
            A21 = a.getSubMatrix(rkn, r1k)
            A22 = a.getSubMatrix(rkn, rkn)
            B11 = b.getSubMatrix(r1k, r1k)
            B12 = b.getSubMatrix(r1k, rkn)
            B21 = b.getSubMatrix(rkn, r1k)
            B22 = b.getSubMatrix(rkn, rkn)
            M1 = MatrixPSM.multiply2(A11 , MatrixPSM.minus(B12, B22))
            M2 = MatrixPSM.multiply2(MatrixPSM.plus(A11, A12) , B22)
            M3 = MatrixPSM.multiply2(MatrixPSM.plus(A21, A22) , B11)
            M4 = MatrixPSM.multiply2(A22 , MatrixPSM.minus(B21, B11))
            M5 = MatrixPSM.multiply2(MatrixPSM.plus(A11, A22) , MatrixPSM.plus(B11, B22))
            M6 = MatrixPSM.multiply2(MatrixPSM.minus(A12, A22) , MatrixPSM.plus(B21, B22))
            M7 = MatrixPSM.multiply2(MatrixPSM.minus(A11, A21), MatrixPSM.plus(B11, B12))
            C11 = MatrixPSM.minus(MatrixPSM.plus(M4, M5, M6) , M2)
            C12 = MatrixPSM.plus(M1, M2)
            C21 = MatrixPSM.plus(M3, M4)
            C22 = MatrixPSM.minus(MatrixPSM.plus(M1, M5) , MatrixPSM.plus(M3, M7))
            return MatrixPSM.joinMatrix(MatrixPSM.joinMatrix(C11, C12, 0), MatrixPSM.joinMatrix(C21, C22, 0), 1)
        
        if False in [isinstance(item, MatrixPSM) for item in [a, b]]:
            raise Exception("�������Ǿ���")
        if a.colNum() != b.rowNum():
            raise Exception("��ģ�����ϣ������������")
        if a.isSquareMatrix() and b.isSquareMatrix() and b.rowNum() % 2 == 0 :
            return multiplySameEvenSquare(a, b)
        else:
            trans_b = b.transpose()
            answer = []
            for i in range(a.rowNum()):
                answer.append([])
            for i in range(a.rowNum()):
                for j in range(trans_b.rowNum()):
                    answer[i].\
                    append(inner_product(a.rawMatrix[i], trans_b.rawMatrix[j]))
            # return answer
            return MatrixPSM([[item * (not(abs(item) < 1E-14))\
                                for item in row] for row in answer])

    @staticmethod
    def multiply(*matrix):
        # ���ݷָ����󣬵ݹ������˻�
        def forSepr(matrix, sepr, m, n):
            if n == m + 1:
                return MatrixPSM.multiply2(matrix[m], matrix[n])
            k = sepr.rawMatrix[m][n]
            if m == k:
                return MatrixPSM.\
                    multiply2(matrix[m], forSepr(matrix, sepr, k + 1, n))
            elif n == k + 1:
                return MatrixPSM.\
                    multiply2(forSepr(matrix, sepr, m, k), matrix[n])
            elif m == n:
                return matrix[m]
            
        if False in [isinstance(item, MatrixPSM) for item in matrix]:
            raise Exception("�������Ǿ���")
    
        # ��������ģ�Ƿ���ȷ
        numOfMat = len(matrix)
        rowSize = map(lambda x:x.rowNum(), matrix)
        colSize = map(lambda x:x.colNum(), matrix)
        rowSize.append(colSize[-1])
        colSize.insert(0, rowSize[0])
        if [colSize[i] == rowSize[i] for i in range(len(rowSize))].\
        count(False) > 0:
            raise  Exception("��ģ�����ϣ������������")
        else :
            # �������۾���cost[2][4]�������m[2]*m[3]*m[4]�ĳ˷�����
            cost = MatrixPSM.zeros(numOfMat, numOfMat)
            # �����ָ�λ�þ���Ԫ��Ϊ0�����ڵ�0������֮��ָ�
            sepr = MatrixPSM.zeros(numOfMat, numOfMat)
            # add�ǶԽ�������������ÿ�μ���һ���Խ���
            for add in range(1, numOfMat):
                for i in range(0, numOfMat - add):
                    j = i + add
                    # ��ʼ�����ֵ
                    cost.rawMatrix[i][j] = 9.999999e100
                    # k�Ǽ���cost[i][j]ʱ�ķָ���
                    # ����m[1]*m[2]*m[3]=m[1]*(m[2]*m[3])ʱ��k=1,���ڵ�һ�������ָ�
                    # ����m[1]*m[2]*m[3]=(m[1]*m[2])*m[3]ʱ��k=2,���ڵڶ��������ָ�
                    for k in range(i , j):
                        currCost = cost.rawMatrix[i][k] \
                        + cost.rawMatrix[k + 1][j] \
                        + rowSize[i] * rowSize[k + 1] * rowSize[j + 1] 
                        if currCost < cost.rawMatrix[i][j]:
                            cost.rawMatrix[i][j] = currCost
                            sepr.rawMatrix[i][j] = k
            # cost.printMatrix();sepr.printMatrix()
            return forSepr(matrix, sepr, 0, numOfMat - 1)
        # return reduce(lambda x, y:MatrixPSM.multiply2(x, y), matrix)

    def inverse(self):
        # matrix = copy.deepcopy(matrix)
        if not self.isSquareMatrix():
            raise Exception("�������Ƿ���")
        length = self.rowNum()
        if self.determinant() == 0:
            raise Exception("����������")
        aug = MatrixPSM.joinMatrix(self, MatrixPSM.eyes(length, length), 0)
        norm_aug = aug.normalize()
        # return getSubMatrix(norm_aug,range(1,length+1),range(length+1,2*length+1))
        a = norm_aug.getSubMatrix(range(length), range(length , 2 * length))
        return MatrixPSM([[item * (not(abs(item) < 1E-15)) \
                           for item in row] \
                          for row in a.rawMatrix])
    
    @staticmethod
    def solveEquation(matrix, b):
        b = copy.deepcopy(b)
        matrix = copy.deepcopy(matrix)
        if False in [isinstance(item, MatrixPSM) for item in [matrix, b]]:
            raise Exception("�������Ǿ���")
        if not MatrixPSM.isRowSizeEqual(matrix, b):
            raise Exception("��ģ�����ϣ�����������ɷ�����")
        aug = MatrixPSM.joinMatrix(matrix, b, 0)
        aug = aug.normalize()
        # print aug
        r_matrix = matrix.rank()
        r_aug = aug.rank()
        if r_matrix < r_aug:
            return None
        else:
            try:
                # ɾ������֮�󣬾����������٣�����IndexError
                for i in range(aug.rowNum()):
                    if inner_product([1] * aug.colNum(), aug.rawMatrix[i]) == 0:
                        del aug.rawMatrix[i]
            except IndexError:
                pass
            return MatrixPSM([aug.transpose().rawMatrix[-1][:r_aug]]).transpose()
    
    

if __name__ == '__main__':

    matrix1 = MatrixPSM([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    matrix2 = MatrixPSM([[2, 2, 1], [3, 7, 4], [2, 6, 8]])
    matrix3 = MatrixPSM([[2, 2, 1], [3, 7, 4], [2, 6, 8]])
    matrix4 = MatrixPSM([[1, 2], [2, 1]])
    
    matrix5 = MatrixPSM([[1, 2, 3, 4], [4, 4, 6, 7], [0, 8, 9, 10], [7, 6, 5, 4]])
    list1 = [1, 2, 3]
    list2 = [2, 1, 0]
    
    MatrixPSM.solveEquation(matrix2, MatrixPSM([[3], [3], [1]])).printMatrix()
    print MatrixPSM.determinant(matrix5)
    MatrixPSM.multiply(matrix5.inverse(), matrix5).printMatrix()
    MatrixPSM.multiply(matrix1, matrix1, matrix1).printMatrix()
    print MatrixPSM.plus(matrix1, matrix2, matrix3)
    
    matrix6 = MatrixPSM([[2, 1], [3, 4], [2, 8]])
    matrix7 = MatrixPSM([[1, 2, 3, 4], [4, 4, 2, 1]])
    matrix8 = MatrixPSM([[1, 3, 4, 3, 4], [4, 4, 1, 3, 4], [0, 1, 2, 5, 4], [ 2, 3, 4, 5, 4]])
    matrix9 = MatrixPSM([[2, 1, 1, 3, 2]]).transpose()
    time1 = time.clock()
    MatrixPSM.multiply(matrix6, matrix7, matrix8, matrix9).printMatrix()
    print time.clock() - time1
    
    print getMinMax(*[1, 1, 4, 2, 7, -3, 9, 55, 8, 8])
    print MatrixPSM([list1]).isSquareMatrix(), matrix1.isSquareMatrix()
    
    time1 = time.clock()
    for i in range(100000):
        matrix1.transpose()
    print time.clock() - time1
    
    b = matrix1.transpose()
    b.rawMatrix[0] = 34
    b.printMatrix()
    matrix1.printMatrix()
    
