'''
Brian's version of scipy sparse matrix support starting from 0.7.1
'''

from base import *
from lil import *

__all__ = filter(lambda s:not s.startswith('_'),dir())

if __name__=='__main__':
    x = lil_matrix((10, 10))
    x[3,5] = 3
    print x.todense()
    