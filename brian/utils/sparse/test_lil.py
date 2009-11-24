from numpy import array
from base import *
from lil import *

def test_lil():
    x = lil_matrix((10, 10))
    test_calls = '''
    x[0, 4] = 3
    x[1, :] = 5
    x[2, []] = []
    x[3, [0]] = [6]
    x[4, [0, 1]] = [7, 8]
    x[5, 2:4] = [1,2]
    x[6:8, 4:6] = 1
    '''
    for line in test_calls.split('\n'):
        line = line.strip()
        if line:
            try:
                exec line
            except Exception, e:
                print 'Line:', line
                print repr(e)
                print

if __name__=='__main__':
    test_lil()