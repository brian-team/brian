'''
**** OUT OF DATE AND NO LONGER USEFUL BECAUSE OF NEW CONNECTIONS ****

This class was originally written because of problems with STDP, with
the new connection class structure, this shouldn't be necessary. In fact
it won't even work with the new class structure.
'''

from brian import *
from brian.connection import ConnectionMatrix, DenseConnectionMatrix
from scipy import sparse
import numpy

__all__ = ['MaskedDenseConnectionMatrix']

class MaskedDenseConnectionMatrix(DenseConnectionMatrix):
    '''
    Sneaky little connection matrix that has dense underlying
    storage, for fast row and column access, but a potentially
    sparse mask. You provide a function rowget(i) to the initialiser
    and the output of this should be a list or array of indices
    for row i of the matrix. An example would be to have a sparse
    matrix, see the sample code in masked_mat.py.
    '''
    def __new__(subtype, dims, rowget=None, maskstructure=None, **kwds):
        return numpy.zeros(dims, dtype=float).view(subtype)
    def __init__(self, dims, rowget=None, maskstructure=None, **kwds):
        if rowget is None and maskstructure is None:
            raise ValueError('Need to specify either rowget or maskstructure')
        DenseConnectionMatrix.__init__(self, dims)
        if maskstructure is not None:
            self.mask = maskstructure(dims, **kwds)
            if rowget is None:
                if maskstructure is sparse.lil_matrix:
                    def rowget(i):
                        return self.mask.rows[i]
                elif maskstructure is sparse.csr_matrix:
                    def rowget(i):
                        x = self.mask
                        return x.indices[x.indptr[i]:x.indptr[i+1]]
                else:
                    raise ValueError('You need to specify rowget for sparse matrices of that type.')
            self.rowget = rowget
        else:
            self.rowget = rowget
    def add_row(self,i,X):
        J = self.rowget(i)
        X[J] += self[i,J]    
    def add_rows(self,spikes,X):
        for i in spikes:
            J = self.rowget(i)
            X[J] += self[i,J]
    def __getitem__(self, item):
        if isinstance(item, tuple) and len(item)==2 and isinstance(item[0],int) and item[1]==slice(None,None,None):
             x = numpy.zeros(self.shape[1])
             J = self.rowget(item[0])
             x[J] = self[item[0], J]
             return x
        return DenseConnectionMatrix.__getitem__(self, item)

if __name__=='__main__':
#    G = NeuronGroup(10, model='V:1')
#    S = SpikeGeneratorGroup(1, [(0,0*ms)])
#    rowget = lambda i : mat.rows[i]
#    C = Connection(S, G, structure=MaskedDenseConnectionMatrix, rowget=rowget)
#    C.W[:] = 1
#    mat = sparse.lil_matrix(C.W.shape, dtype=bool)
#    mat[0,1] = True
#    run(1*ms)
#    print G.V
    G = NeuronGroup(10, model='V:1')
    S = SpikeGeneratorGroup(1, [(0,0*ms)])
    C = Connection(S, G, structure=MaskedDenseConnectionMatrix,
                   maskstructure=sparse.csr_matrix)
    C.W[:] = 1
    C.W.mask[0,1] = True
    run(1*ms)
    print G.V
    print C.W[0,:]
    print asarray(C.W)[0,:]