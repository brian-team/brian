# NOTE: this is now in Brian.connections

from brian import *
from brian.connection import ConnectionMatrix
import random as pyrandom
from scipy import random as scirandom
import numpy

__all__ = ['UserComputedConnectionMatrix', 'UserComputedSparseConnectionMatrix', 'random_row_func', 'random_sparse_row_func']

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


class UserComputedConnectionMatrix(ConnectionMatrix):
    '''
    A computed connection matrix defined by a user-specified function
    
    Normally this matrix will be initialised by passing the class
    object to the :class:`Connection` object. In the initialisation
    of the :class:`Connection` specify ``structure=UserComputedConnectionMatrix``
    and add the keyword ``row_func=...``, e.g.::
    
        def f(i):
            return max_weight*ones(N)/(1+(arange(N)-i)**2)
        C = Connection(G1, G2, structure=UserComputedConnectionMatrix, row_func=f)
    
    Initialisation arguments:
    
    ``dims``
        The pair ``(N,M)`` specifying the dimensions of the matrix.
    ``row_func``
        The function ``f(i)`` which returns an array of length ``M``,
        the weight matrix for row ``i``. Note that you are responsible
        for making sure the function returns consistent results (so
        random functions should be initialised with a seed based on
        the row ``i``).
    
    **Limitations**
    
    This type of connection matrix cannot be changed during a run, and
    cannot be used with methods like :class:`Connection.connect_random`.
    
    **Efficiency considerations**
    
    This connection matrix is for dense connectivity, if the connectivity
    is sparse you might get better performance with :class:`UserComputedSparseConnectionMatrix`.
    '''
    def __init__(self, dims, row_func):
        self.sourcelen, self.targetlen = dims
        self.row_func = row_func

    def get_row(self, i):
        return self.row_func(i)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.get_row(item)
        if isinstance(item, tuple):
            if len(item) == 2:
                item_i, item_j = item
                if isinstance(item_i, int) and isinstance(item_j, slice):
                    if is_colon_slice(item_j):
                        return self.get_row(item_i)
        raise ValueError('Only "i,:" indexing supported.')

def random_sparse_row_func(N, p, weight=1., initseed=None):
    '''
    Returns a random connectivity ``row_func`` for use with :class:`UserComputedSparseConnectionMatrix`
    
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
    myrange = numpy.arange(N, dtype=int)
    def row_func(i):
        pyrandom.seed(initseed + int(i))
        scirandom.seed(initseed + int(i))
        k = scirandom.binomial(N, p, 1)[0]
        return (pyrandom.sample(myrange, k), weight)
    return row_func


class UserComputedSparseConnectionMatrix(ConnectionMatrix):
    '''
    A computed sparse connection matrix defined by a user-specified function
    
    Normally this matrix will be initialised by passing the class
    object to the :class:`Connection` object. In the initialisation
    of the :class:`Connection` specify ``structure=UserComputedSparseConnectionMatrix``
    and add the keyword ``row_func=...``, e.g.::
    
        def f(i):
            if 0<i<N-1:
                return ([i-1,i+1], weight*ones(2))
            elif i>0:
                return ([i-1], weight*ones(1))
            else:
                return ([i+1], weight*ones(1))
        C = Connection(G1, G2, structure=UserComputedSparseConnectionMatrix, row_func=f)
    
    Initialisation arguments:
    
    ``dims``
        The pair ``(N,M)`` specifying the dimensions of the matrix.
    ``row_func``
        The function ``f(i)`` which for a row ``i`` returns a pair ``(indices, values))``
        consisting of a list or array ``indices`` with the indices of the
        nonzero elements of the row, and an array of the same length ``values``
        giving the weight matrix for those indices. Note that you are responsible
        for making sure the function returns consistent results (so
        random functions should be initialised with a seed based on
        the row ``i``).
    
    **Limitations**
    
    This type of connection matrix cannot be changed during a run, and
    cannot be used with methods like :class:`Connection.connect_random`.
    
    **Efficiency considerations**
    
    This connection matrix is for sparse connectivity, if the connectivity
    is dense you might get better performance with :class:`UserComputedConnectionMatrix`.
    '''
    def __init__(self, dims, row_func):
        self.sourcelen, self.targetlen = dims
        self.row_func = row_func
        self.cur_row = numpy.zeros(dims[1])

    def add_row(self, i, X):
        indices, values = self.row_func(i)
        X[indices] += values

    def add_scaled_row(self, i, X, factor):
        # modulation may not work? need factor[self.rows[i]] here? is factor a number or an array?
        X[indices] += factor * values

    def get_row(self, i):
        indices, values = self.row_func(i)
        self.cur_row[:] = 0.0
        self.cur_row[indices] = values
        return self.cur_row

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.get_row(item)
        if isinstance(item, tuple):
            if len(item) == 2:
                item_i, item_j = item
                if isinstance(item_i, int) and isinstance(item_j, slice):
                    if is_colon_slice(item_j):
                        return self.get_row(item_i)
        raise ValueError('Only "i,:" indexing supported.')

if __name__ == '__main__':

    eqs = '''
    dv/dt = (ge+gi-(v+49*mV))/(20*ms) : volt
    dge/dt = -ge/(5*ms) : volt
    dgi/dt = -gi/(10*ms) : volt
    '''

    N = 4000
    #N = 20000
    #N = 100000
    #N = 1000000
    duration = 1000 * ms
    usecc = True
    usesparsecc = True

    Ne = int(0.8 * N)
    Ni = N - Ne
    prob = 80. / N

    P = NeuronGroup(N, model=eqs, threshold= -50 * mV, reset= -60 * mV)
    P.v = -60 * mV + 10 * mV * rand(len(P))
    Pe = P.subgroup(Ne)
    Pi = P.subgroup(Ni)

    if usecc:
        if usesparsecc:
            connmat = UserComputedSparseConnectionMatrix
            rfunc = random_sparse_row_func
        else:
            connmat = UserComputedConnectionMatrix
            rfunc = random_row_func
        Ce = Connection(Pe, P, 'ge', structure=connmat,
                      row_func=rfunc(N, p=prob, weight=1.62 * mV))
        Ci = Connection(Pi, P, 'gi', structure=connmat,
                      row_func=rfunc(N, p=prob, weight= -9 * mV))
    else:
        Ce = Connection(Pe, P, 'ge')
        Ci = Connection(Pi, P, 'gi')
        Ce.connect_random(Pe, P, prob, weight=1.62 * mV)
        Ci.connect_random(Pi, P, prob, weight= -9 * mV)

    if N > 5000:
        M = PopulationSpikeCounter(P)
    else:
        M = SpikeMonitor(P)

    if N > 100000:
        @network_operation()
        def f(t):
            print 'Current time', t

    import time
    t = time.time()
    run(duration)
    if usecc:
        if usesparsecc:
            print 'Using sparse computed connection'
        else:
            print 'Using computed connection'
    else:
        print 'Using standard connection'
    print "Time taken to run simulation:", time.time() - t
    print "Number of spikes:", M.nspikes
    if N <= 5000:
        raster_plot(M)
        show()

# Statistics, sample runs on my laptop
#
#N=4000, T=1s
#
#Using standard connection
#Time taken to run simulation: 5.76499986649
#Number of spikes: 24860
#
#Using computed connection
#Time taken to run simulation: 12.0629999638
#Number of spikes: 24761
#
#Using sparse computed connection
#Time taken to run simulation: 11.3589999676
#Number of spikes: 23421
#
#N=20000, T=1s
#
#Using standard connection
#Time taken to run simulation: 23.8900001049
#Number of spikes: 122241
#RAM usage: 94MB
#
#Using computed connection
#Time taken to run simulation: 74.75
#Number of spikes: 121995
#RAM usage: 39MB
#
#Using sparse computed connection
#Time taken to run simulation: 49.3440001011
#Number of spikes: 121841
#RAM usage: 39MB
#
#N=100000, T=10ms
#
#Using standard connection
#Time taken to run simulation: 8.17100000381
#Number of spikes: 10971
#RAM usage: 330MB
#
#Using computed connection
#Time taken to run simulation: 13.390999794
#Number of spikes: 11299
#RAM usage: 45MB
#
#Using sparse computed connection
#Time taken to run simulation: 4.98399996758
#Number of spikes: 11035
#RAM usage: 45MB
#
#N=1m, T=1ms
#
#Using computed connection
#Time taken to run simulation: 137.796999931
#Number of spikes: 6219
#RAM usage: 95MB
#
#Using sparse computed connection
#Time taken to run simulation: 4.04700016975
#Number of spikes: 6181
#RAM usage: 120MB
