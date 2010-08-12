from base import *
from sparsematrix import *

__all__ = ['random_row_func', 'random_matrix',
           'random_matrix_fixed_column', 'eye_lil_matrix',
           ]

def random_row_func(N, p, weight=1., initseed=None):
    '''
    Returns a random connectivity ``row_func`` for use with :class:`UserComputedConnectionMatrix`
    
    Gives equivalent output to the :meth:`Connection.connect_random` method.
    
    Arguments:
    
    ``N``
        The number of target neurons.
    ``p``
        The probability of a synapse.
    ``weight``
        The connection weight (must be a single value).
    ``initseed``
        The initial seed value (for reproducible results).
    '''
    if initseed is None:
        initseed = pyrandom.randint(100000, 1000000) # replace this
    cur_row = numpy.zeros(N)
    myrange = numpy.arange(N, dtype=int)

    def row_func(i):
        pyrandom.seed(initseed + int(i))
        scirandom.seed(initseed + int(i))
        k = scirandom.binomial(N, p, 1)[0]
        cur_row[:] = 0.0
        cur_row[pyrandom.sample(myrange, k)] = weight
        return cur_row

    return row_func


# Generation of matrices
def random_matrix(n, m, p, value=1.):
    '''
    Generates a sparse random matrix with size (n,m).
    Entries are 1 (or optionnally value) with probability p.
    If value is a function, then that function is called for each
    non zero element as value() or value(i,j).
    '''
    # TODO:
    # Simplify (by using valuef)
    W = sparse.lil_matrix((n, m))
    if callable(value) and callable(p):
        if value.func_code.co_argcount == 0:
            valuef = lambda i, j:[value() for _ in j] # value function
        elif value.func_code.co_argcount == 2:
            try:
                failed = (array(value(0, arange(m))).size != m)
            except:
                failed = True
            if failed: # vector-based not possible
                log_debug('connections', 'Cannot build the connection matrix by rows')
                valuef = lambda i, j:[value(i, k) for k in j]
            else:
                valuef = value
        else:
            raise AttributeError, "Bad number of arguments in value function (should be 0 or 2)"

        if p.func_code.co_argcount == 2:
            # Check if p(i,j) is vectorisable
            try:
                failed = (array(p(0, arange(m))).size != m)
            except:
                failed = True
            if failed: # vector-based not possible
                log_debug('connections', 'Cannot build the connection matrix by rows')
                for i in xrange(n):
                    W.rows[i] = [j for j in range(m) if rand() < p(i, j)]
                    W.data[i] = list(valuef(i, array(W.rows[i])))
            else: # vector-based possible
                for i in xrange(n):
                    W.rows[i] = list((rand(m) < p(i, arange(m))).nonzero()[0])
                    W.data[i] = list(valuef(i, array(W.rows[i])))
        elif p.func_code.co_argcount == 0:
            for i in xrange(n):
                W.rows[i] = [j for j in range(m) if rand() < p()]
                W.data[i] = list(valuef(i, array(W.rows[i])))
        else:
            raise AttributeError, "Bad number of arguments in p function (should be 2)"
    elif callable(value):
        if value.func_code.co_argcount == 0: # TODO: should work with partial objects
            for i in xrange(n):
                k = random.binomial(m, p, 1)[0]
                W.rows[i] = sample(xrange(m), k)
                W.rows[i].sort()
                W.data[i] = [value() for _ in xrange(k)]
        elif value.func_code.co_argcount == 2:
            try:
                failed = (array(value(0, arange(m))).size != m)
            except:
                failed = True
            if failed: # vector-based not possible
                log_debug('connections', 'Cannot build the connection matrix by rows')
                for i in xrange(n):
                    k = random.binomial(m, p, 1)[0]
                    W.rows[i] = sample(xrange(m), k)
                    W.rows[i].sort()
                    W.data[i] = [value(i, j) for j in W.rows[i]]
            else:
                for i in xrange(n):
                    k = random.binomial(m, p, 1)[0]
                    W.rows[i] = sample(xrange(m), k)
                    W.rows[i].sort()
                    W.data[i] = list(value(i, array(W.rows[i])))
        else:
            raise AttributeError, "Bad number of arguments in value function (should be 0 or 2)"
    elif callable(p):
        if p.func_code.co_argcount == 2:
            # Check if p(i,j) is vectorisable
            try:
                failed = (array(p(0, arange(m))).size != m)
            except:
                failed = True
            if failed: # vector-based not possible
                log_debug('connections', 'Cannot build the connection matrix by rows')
                for i in xrange(n):
                    W.rows[i] = [j for j in range(m) if rand() < p(i, j)]
                    W.data[i] = [value] * len(W.rows[i])
            else: # vector-based possible
                for i in xrange(n):
                    W.rows[i] = list((rand(m) < p(i, arange(m))).nonzero()[0])
                    W.data[i] = [value] * len(W.rows[i])
        elif p.func_code.co_argcount == 0:
            for i in xrange(n):
                W.rows[i] = [j for j in range(m) if rand() < p()]
                W.data[i] = [value] * len(W.rows[i])
        else:
            raise AttributeError, "Bad number of arguments in p function (should be 2)"
    else:
        for i in xrange(n):
            k = random.binomial(m, p, 1)[0]
            # Not significantly faster to generate all random numbers in one pass
            # N.B.: the sample method is implemented in Python and it is not in Scipy
            W.rows[i] = sample(xrange(m), k)
            W.rows[i].sort()
            W.data[i] = [value] * k

    return W

def random_matrix_fixed_column(n, m, p, value=1.):
    '''
    Generates a sparse random matrix with size (n,m).
    Entries are 1 (or optionnally value) with probability p.
    The number of non-zero entries by per column is fixed: (int)(p*n)
    If value is a function, then that function is called for each
    non zero element as value() or value(i,j).
    '''
    W = sparse.lil_matrix((n, m))
    k = (int)(p * n)
    for j in xrange(m):
        # N.B.: the sample method is implemented in Python and it is not in Scipy
        for i in sample(xrange(n), k):
            W.rows[i].append(j)

    if callable(value):
        if value.func_code.co_argcount == 0:
            for i in xrange(n):
                W.data[i] = [value() for _ in xrange(len(W.rows[i]))]
        elif value.func_code.co_argcount == 2:
            for i in xrange(n):
                W.data[i] = [value(i, j) for j in W.rows[i]]
        else:
            raise AttributeError, "Bad number of arguments in value function (should be 0 or 2)"
    else:
        for i in xrange(n):
            W.data[i] = [value] * len(W.rows[i])

    return W

def eye_lil_matrix(n):
    '''
    Returns the identity matrix of size n as a lil_matrix
    (sparse matrix).
    '''
    M = sparse.lil_matrix((n, n))
    M.setdiag([1.] * n)
    return M
