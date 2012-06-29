'''
This package maintains multiple patches for scipy.sparse.lil_matrix for the
various different versions of scipy. It chooses dynamically at run time which
version to use.
'''
import scipy, warnings, os, imp
import scipy.sparse

__all__ = ['lil_matrix', 'spmatrix']

default = '0_7_1'
table = {
    '0.10.1':    '0_7_1',
    '0.10.0':    '0_7_1',
    '0.9.0':     '0_7_1',
    '0.8.0':     '0_7_1',
    '0.7.2':     '0_7_1',
    '0.7.1':     '0_7_1',
    '0.7.1rc3':  '0_7_1',
    '0.7.1rc2':  '0_7_1',
    '0.7.1rc1':  '0_7_1',
    '0.7.0':     '0_7_0',
    '0.7.0rc2':  '0_7_0',
    '0.7.0b1':   '0_7_0',
    '0.6.0':     '0_6_0',
    }

if scipy.__version__ in table:
    modulename = table[scipy.__version__]
else:
    for i in xrange(1, len(scipy.__version__)):
        v = scipy.__version__[:-i]
        if v in table:
            modulename = table[v]
            break
    else:
        modulename = default
        warnings.warn("Couldn't find matching sparse matrix patch for scipy version %s, but in most cases this shouldn't be a problem." % scipy.__version__)

module = __import__(modulename, globals(), locals(), [], -1)
lil_matrix = module.lil_matrix
spmatrix = scipy.sparse.spmatrix

if __name__ == '__main__':
    x = lil_matrix((5, 5))
    x[2, 3] = 5
    print x.todense()

