# -*- coding: cp936 -*-
# 输入两个n位二进制整数，输出其乘积
import copy
import time
'''矩阵计算的模块

注：矩阵一定是二维的，矩阵的函数不接受列表
矩阵的行列序号都以0开始
'''

def isVector(li):
    u"""数字不是向量，矩阵不是向量"""
    if not isinstance(li, list):
        return False
    n = 0
    for i in range(len(li)):
        if not isinstance(li[i], list):
            n += 1
    return True if len(li) == n else False

def inner_product(a, b):
    if isVector(a) and isVector(b):
        if len(a) != len(b):
            raise Exception("两个向量规模不同")
        return sum(map(lambda i, j:i * j, a, b))
    else:
        raise Exception("参数不是向量")




class MatrixPSM:
    def __init__(self, m):
        self.rawMatrix = m

    def rowNum(self):
        return len(self.rawMatrix)
    def colNum(self):
        return len(self.rawMatrix[0])
       
    @staticmethod
    def isRawMatrix(li):
        u'''判断list是否是矩阵'''
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
        # 列表解析速度更快一点
        # return map(lambda *x:list(x), *self.rawMatrix)
        return MatrixPSM([[row[i] for row in self.rawMatrix] \
                          for i in range(len(self.rawMatrix[0]))])

    @staticmethod
    def isRowSizeEqual(*matrix):
        if map(lambda item:item.rowNum(), matrix)\
        .count(matrix[0].rowNum()) != len(matrix):
            return False
        else:
            return True
    @staticmethod
    def isColumnSizeEqual(*matrix):
        if map(lambda item:item.colNum(), matrix)\
        .count(matrix[0].colNum()) != len(matrix):
            return False
        else:
            return True
    @staticmethod
    def isSizeEqual(*matrix):
        if False in [isinstance(item, MatrixPSM) for item in matrix]:
            raise Exception("参数不是矩阵！")
        elif not (MatrixPSM.isRowSizeEqual(*matrix) \
                  and MatrixPSM.isColumnSizeEqual(*matrix)):
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
        u'''把一个或多个列表转化为矩阵
        ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！不完善
        '''
        length = map(lambda x:len(x), tupl)
        if length.count(len(tupl[0])) == len(tupl):
            return MatrixPSM(list(tupl))
        else :
            raise Exception("参数规模不同")
        
    def printMatrix(self):
        print "["
        for i in self.rawMatrix:
            print " ", i
        print "]"
    
    def getSubMatrix(self, row, column):
        # matrix = copy.deepcopy(matrix)
        u'''返回row中的行，column中的列组成的子矩阵
            
        '''
        if row == [] or column == []:
            return MatrixPSM([[]])
        try:
            return MatrixPSM([[self.rawMatrix[r][c] \
                               for c in column] for r in row])
            # return transpose(map(lambda y:transpose(map(lambda x:matrix[x], row))[y], column))
        except IndexError:
            raise Exception("参数错误")
    
    @staticmethod
    def joinMatrix(m1, m2, direction):
        u'''连接两个矩阵
        direction是0，则横向连接，是1则纵向连接'''
        if False in [isinstance(item, MatrixPSM) for item in [m1, m2]]:
            raise Exception("参数不是矩阵！")
        if direction == 0:
            if m1.rowNum() != m2.rowNum():
                raise Exception("两个矩阵规模不同，不能横向连接")
            else:
                return MatrixPSM(map(\
                                     lambda x, y:x + y, \
                                     m1.rawMatrix, m2.rawMatrix))
        elif direction == 1:
            if m1.colNum() != m2.colNum():
                raise Exception("两个矩阵规模不同，不能纵向连接")
            else:
                return MatrixPSM(m1.rawMatrix + m2.rawMatrix)
        else:
            raise Exception("参数direction错误，只能为0或1")
    
    @staticmethod
    def zeros(m, n):
        u'''生成矩阵，元素全为0'''
        return MatrixPSM([[0 for i in range(n)] for i in range(m)])
    @staticmethod
    def eyes(m, n):
        u'''生成单位矩阵'''
        a = MatrixPSM.zeros(m, n)
        for i in range(min(m, n)):
            a.rawMatrix[i][i] = 1
        return a
    @staticmethod
    def ones(m, n):
        u'''生成矩阵，元素全为1'''
        return MatrixPSM([[1 for i in range(n)] for i in range(m)])
    

    
    def diag(self):
        # matrix = copy.deepcopy(matrix)
        u'''获取对角线上的元素，组成一个行矩阵
        
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
        u'''初等行变换对角化'''
        a = copy.deepcopy(self.rawMatrix)
        for j in range(min(len(a), len(a[0]))):
            if a[j][j] == 0:
                for i in range(j + 1, len(a)):
                    if a[i][j] != 0:
                        a[j], a[i] = a[i], a[j]
                        # 换行后，把一行加负号
                        for i in range(len(a[j])):
                            a[1][i] = -a[1][i]         
                        break
            if a[j][j] == 0:
                return a
            for i in range(j):
                a[i] = list(map(\
                        lambda x, y:x - y * a[i][j] / float(a[j][j]), \
                         a[i], a[j]))
            for i in range(j + 1, len(a)):
                a[i] = list(map(\
                        lambda x, y:x - y * a[i][j] / float(a[j][j]), \
                        a[i], a[j]))
        return MatrixPSM(a)
    
    def normalize(self):
        # matrix = copy.deepcopy(self.rawMatrix)
        u'''初等行变换化标准形'''
        diag = self.diagonalize()
        for i in range(min(self.rowNum(), self.colNum())):
            if diag.rawMatrix[i][i] != 0:
                diag.rawMatrix[i] = \
                [item / float(diag.rawMatrix[i][i]) \
                 for item in diag.rawMatrix[i]]
        return diag
    
    def rank(self):
        # matrix = copy.deepcopy(matrix)
        u'''求矩阵的秩'''
        diag = self.diagonalize()
        rank = 0
        for i in range(len(self.rawMatrix)):
            if inner_product([1] * self.colNum(), diag.rawMatrix[i]) != 0:
                rank += 1
        return rank
    
    def determinant(self):
        # matrix = copy.deepcopy(matrix)
        u'''求行列式值'''
        if not self.isSquareMatrix():
            raise Exception("参数不是方阵")
        diag = self.diagonalize()
        b = 1
        for i in range(diag.rowNum()):
            b *= diag.rawMatrix[i][i]
        return b
    
    @staticmethod
    def plus(*matrix):
        u'''矩阵相加'''
        if False in [isinstance(item, MatrixPSM) for item in matrix]:
            raise Exception("参数不是矩阵！")
        if not MatrixPSM.isSizeEqual(*matrix):
            raise Exception("参数规模不同")
        return MatrixPSM(map(\
                lambda *row:map(lambda *column:sum(column), *row),
                *[mat.rawMatrix for mat in matrix]))
    @staticmethod
    def minus(a, b):
        u'''矩阵相减'''
        if False in [isinstance(item, MatrixPSM) for item in [a, b]]:
            raise Exception("参数不是矩阵！")
        if not MatrixPSM.isSizeEqual(a, b):
            raise Exception("参数规模不同")
        return MatrixPSM(map(\
                lambda x, y:map(lambda i, j:i - j, x, y), \
                a.rawMatrix, b.rawMatrix))


    @staticmethod
    def expand_size2power_of_2(a, max_size=-1):
        # 把矩阵规模扩大到2的幂阶方阵，若指定了max_size，则扩张后的矩阵规模大于等于max_size
        rowNum = a.rowNum()
        colNum = a.colNum()
        
        if rowNum == 0 and colNum == 0:
            return a
        if rowNum == 1 and colNum == 1:
            return a
        
        if max_size == -1:
            max_size = max(rowNum, colNum)
        
        i = -1
        while True:
            i += 1
            if max_size <= pow(2, i):
                break
        need_size = pow(2, i)
        
        row_lack = need_size - rowNum
        col_lack = need_size - colNum
        if row_lack > 0:
            a = MatrixPSM.joinMatrix(a, MatrixPSM.zeros(row_lack, colNum), 1)
        if col_lack > 0:
            a = MatrixPSM.joinMatrix(a, \
                                     MatrixPSM.zeros(need_size, col_lack), \
                                     0)
        return a
    
    @staticmethod
    def multiply2(a, b):
        # 求两个矩阵乘积
        def multiply_2level(a, b):
            trans_b = b.transpose()
            answer = []
            for i in range(a.rowNum()):
                answer.append([])
            for i in range(a.rowNum()):
                for j in range(trans_b.rowNum()):
                    answer[i].\
                    append(inner_product(a.rawMatrix[i], \
                                         trans_b.rawMatrix[j]))
            # return answer
            return MatrixPSM([[item * (not(abs(item) < 1E-14))\
                                for item in row] for row in answer])
        
        def multiply_powerof2_square(a, b):
            # Strassen矩阵乘法
            n = a.rowNum()
            if n == 1:
                return a.rawMatrix[0][0] * b.rawMatrix[0][0]
            if n == 2:
                return multiply_2level(a, b)
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
            M1 = multiply_powerof2_square(A11 , MatrixPSM.minus(B12, B22))
            M2 = multiply_powerof2_square(MatrixPSM.plus(A11, A12) , B22)
            M3 = multiply_powerof2_square(MatrixPSM.plus(A21, A22) , B11)
            M4 = multiply_powerof2_square(A22 , MatrixPSM.minus(B21, B11))
            M5 = multiply_powerof2_square(MatrixPSM.plus(A11, A22) , \
                                          MatrixPSM.plus(B11, B22))
            M6 = multiply_powerof2_square(MatrixPSM.minus(A12, A22) , \
                                          MatrixPSM.plus(B21, B22))
            M7 = multiply_powerof2_square(MatrixPSM.minus(A11, A21), \
                                          MatrixPSM.plus(B11, B12))
            C11 = MatrixPSM.minus(MatrixPSM.plus(M4, M5, M6) , M2)
            C12 = MatrixPSM.plus(M1, M2)
            C21 = MatrixPSM.plus(M3, M4)
            C22 = MatrixPSM.minus(MatrixPSM.plus(M1, M5) , \
                                  MatrixPSM.plus(M3, M7))
            return MatrixPSM.joinMatrix(MatrixPSM.joinMatrix(C11, C12, 0), \
                                        MatrixPSM.joinMatrix(C21, C22, 0), 1)
        
        if False in [isinstance(item, MatrixPSM) for item in [a, b]]:
            raise Exception("参数不是矩阵！")
        if a.colNum() != b.rowNum():
            raise Exception("规模不符合，参数不能相乘")
        
        max_size = max(a.rowNum(), b.rowNum(), a.colNum(), b.colNum())
        c = MatrixPSM.expand_size2power_of_2(a, max_size)
        d = MatrixPSM.expand_size2power_of_2(b, max_size)
        return multiply_powerof2_square(c, d)\
            .getSubMatrix(range(a.rowNum()), range(b.colNum()))


    @staticmethod
    def multiply(*matrix):
        # 根据分隔矩阵，递归求多个矩阵乘积
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
            raise Exception("参数不是矩阵！")
    
        # 检查参数规模是否正确
        numOfMat = len(matrix)
        rowSize = map(lambda x:x.rowNum(), matrix)
        colSize = map(lambda x:x.colNum(), matrix)
        rowSize.append(colSize[-1])
        colSize.insert(0, rowSize[0])
        if [colSize[i] == rowSize[i] for i in range(len(rowSize))].\
        count(False) > 0:
            raise  Exception("规模不符合，参数不能相乘")
        else :
            # 创建代价矩阵，cost[2][4]代表计算m[2]*m[3]*m[4]的乘法次数
            cost = MatrixPSM.zeros(numOfMat, numOfMat)
            # 创建分割位置矩阵，元素为0代表在第0个矩阵之后分隔
            sepr = MatrixPSM.zeros(numOfMat, numOfMat)
            # add是对角线条数，程序每次计算一条对角线
            for add in range(1, numOfMat):
                for i in range(0, numOfMat - add):
                    j = i + add
                    # 初始化最大值
                    cost.rawMatrix[i][j] = 9.999999e100
                    # k是计算cost[i][j]时的分隔处
                    # 例如m[1]*m[2]*m[3]=m[1]*(m[2]*m[3])时，k=1,即在第一个矩阵后分隔
                    # 例如m[1]*m[2]*m[3]=(m[1]*m[2])*m[3]时，k=2,即在第二个矩阵后分隔
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
            raise Exception("参数不是方阵")
        length = self.rowNum()
        if self.determinant() == 0:
            raise Exception("参数不可逆")
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
            raise Exception("参数不是矩阵！")
        if not MatrixPSM.isRowSizeEqual(matrix, b):
            raise Exception("规模不符合，参数不能组成方程组")
        aug = MatrixPSM.joinMatrix(matrix, b, 0)
        aug = aug.normalize()
        # print aug
        r_matrix = matrix.rank()
        r_aug = aug.rank()
        if r_matrix < r_aug:
            return None
        else:
            try:
                # 删除零行之后，矩阵行数变少，导致IndexError
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
    
    
    print MatrixPSM([list1]).isSquareMatrix(), matrix1.isSquareMatrix()
    
    time1 = time.clock()
    for i in range(100000):
        matrix1.transpose()
    print time.clock() - time1
    
    b = matrix1.transpose()
    b.rawMatrix[0] = 34
    b.printMatrix()
    matrix1.printMatrix()
    

