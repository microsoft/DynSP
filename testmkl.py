import numpy as np
import dynet as dt

def test_eigenvalue():
    i= 5000
    data = np.random.random((i,i))
    result = np.linalg.eig(data)
    print (result)

def multi():
    i= 100
    j= 100
    model = dt.Model()
    pA = model.add_parameters((i,j))
    pB = model.add_parameters((j,1))
    A = dt.parameter(pA)
    B = dt.parameter(pB)
    for j in range(1000000):
        result = A * B
        len(result.value())

def npmulti(i,j):
    A = np.random.random((i,j))
    B = np.random.random((i,j))
    result = A * B
    '''
    for p in xrange(1000000):
        result = A * B
        len(result)
    '''

if __name__ == '__main__':
    #test_eigenvalue()
    multi()
    npmulti(5000,5000)
