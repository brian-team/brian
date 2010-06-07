# -*- coding:utf-8 -*-
#class UserComputedConnectionMatrix(ConnectionMatrix):
#    '''
#    A computed connection matrix defined by a user-specified function
#    
#    Normally this matrix will be initialised by passing the class
#    object to the :class:`Connection` object. In the initialisation
#    of the :class:`Connection` specify ``structure=UserComputedConnectionMatrix``
#    and add the keyword ``row_func=...``, e.g.::
#    
#        def f(i):
#            return max_weight*ones(N)/(1+(arange(N)-i)**2)
#        C = Connection(G1, G2, structure=UserComputedConnectionMatrix, row_func=f)
#    
#    Initialisation arguments:
#    
#    ``dims``
#        The pair ``(N,M)`` specifying the dimensions of the matrix.
#    ``row_func``
#        The function ``f(i)`` which returns an array of length ``M``,
#        the weight matrix for row ``i``. Note that you are responsible
#        for making sure the function returns consistent results (so
#        random functions should be initialised with a seed based on
#        the row ``i``).
#    
#    **Limitations**
#    
#    This type of connection matrix cannot be changed during a run, and
#    cannot be used with methods like :class:`Connection.connect_random`.
#    
#    **Efficiency considerations**
#    
#    This connection matrix is for dense connectivity, if the connectivity
#    is sparse you might get better performance with :class:`UserComputedSparseConnectionMatrix`.
#    '''
#    def __init__(self, dims, row_func):
#        self.sourcelen, self.targetlen = dims
#        self.row_func = row_func
#        
#    def get_row(self, i):
#        return self.row_func(i)
#    
#    def __getitem__(self, item):
#        if isinstance(item,int):
#            return self.get_row(item)
#        if isinstance(item,tuple):
#            if len(item)==2:
#                item_i, item_j = item
#                if isinstance(item_i, int) and isinstance(item_j, slice):
#                    if is_colon_slice(item_j):
#                        return self.get_row(item_i)
#        raise ValueError('Only "i,:" indexing supported.')
#
#def random_sparse_row_func(N, p, weight=1., initseed=None):
#    '''
#    Returns a random connectivity ``row_func`` for use with :class:`UserComputedSparseConnectionMatrix`
#    
#    Gives equivalent output to the :meth:`Connection.connect_random` method.
#    
#    Arguments:
#    
#    ``N``
#        The number of target neurons.
#    ``p``
#        The probability of a synapse.
#    ``weight``
#        The connection weight (must be a single value).
#    ``initseed``
#        The initial seed value (for reproducible results).
#    '''
#    if initseed is None:
#        initseed = pyrandom.randint(100000,1000000) # replace this
#    myrange = numpy.arange(N, dtype=int)
#    def row_func(i):
#        pyrandom.seed(initseed+int(i))
#        scirandom.seed(initseed+int(i))
#        k = scirandom.binomial(N, p, 1)[0]
#        return (pyrandom.sample(myrange,k), weight)
#    return row_func
#
#class UserComputedSparseConnectionMatrix(ConnectionMatrix):
#    '''
#    A computed sparse connection matrix defined by a user-specified function
#    
#    Normally this matrix will be initialised by passing the class
#    object to the :class:`Connection` object. In the initialisation
#    of the :class:`Connection` specify ``structure=UserComputedSparseConnectionMatrix``
#    and add the keyword ``row_func=...``, e.g.::
#    
#        def f(i):
#            if 0<i<N-1:
#                return ([i-1,i+1], weight*ones(2))
#            elif i>0:
#                return ([i-1], weight*ones(1))
#            else:
#                return ([i+1], weight*ones(1))
#        C = Connection(G1, G2, structure=UserComputedSparseConnectionMatrix, row_func=f)
#    
#    Initialisation arguments:
#    
#    ``dims``
#        The pair ``(N,M)`` specifying the dimensions of the matrix.
#    ``row_func``
#        The function ``f(i)`` which for a row ``i`` returns a pair ``(indices, values))``
#        consisting of a list or array ``indices`` with the indices of the
#        nonzero elements of the row, and an array of the same length ``values``
#        giving the weight matrix for those indices. Note that you are responsible
#        for making sure the function returns consistent results (so
#        random functions should be initialised with a seed based on
#        the row ``i``).
#    
#    **Limitations**
#    
#    This type of connection matrix cannot be changed during a run, and
#    cannot be used with methods like :class:`Connection.connect_random`.
#    
#    **Efficiency considerations**
#    
#    This connection matrix is for sparse connectivity, if the connectivity
#    is dense you might get better performance with :class:`UserComputedConnectionMatrix`.
#    '''
#    def __init__(self, dims, row_func):
#        self.sourcelen, self.targetlen = dims
#        self.row_func = row_func
#        self.cur_row = numpy.zeros(dims[1])
#        
#    def add_row(self,i,X):
#        indices, values = self.row_func(i)
#        X[indices]+=values
#        
#    def add_scaled_row(self,i,X,factor):
#        # modulation may not work? need factor[self.rows[i]] here? is factor a number or an array?
#        X[indices]+=factor*values
#        
#    def get_row(self, i):
#        indices, values = self.row_func(i)
#        self.cur_row[:] = 0.0
#        self.cur_row[indices] = values
#        return self.cur_row
#    
#    def __getitem__(self, item):
#        if isinstance(item,int):
#            return self.get_row(item)
#        if isinstance(item,tuple):
#            if len(item)==2:
#                item_i, item_j = item
#                if isinstance(item_i, int) and isinstance(item_j, slice):
#                    if is_colon_slice(item_j):
#                        return self.get_row(item_i)
#        raise ValueError('Only "i,:" indexing supported.')
