from brian import *
from brian.connection import ConnectionMatrix
import numpy, scipy
from scipy import weave

__all__ = ['SparseSTDPConnectionMatrix']

colon_slice = slice(None,None,None)

# Sparse matrix with facilities for fast row and column get and set
# - Must be initialised with a completed lil_matrix
# - get methods return numpy arrays
#    . get methods return an array of the complete dimension of
#      the underlying matrix
#    . get methods always return the same array, but filled with
#      different data, so copy it if you need to call a get
#      method more than once
# - set methods perform a masked set operation on a numpy array
#   values where the sparse matrix has no entry are left without
#   an entry, other values are copied
#    . the array passed to set should have the full dimensions
#      of the underlying matrix
# __getitem__, etc. just do what the get/set methods do
#
# Another possibility for a sparse matrix data structure is a 2D doubly
# linked list as mentioned at http://www.math.uu.nl/people/bisselin/PSC/psc4_2.pdf
# the benefit is that insertion and removal of elements is O(1), and
# row and column extraction O(N). This would be useful for people who
# wanted to try doing models of development perhaps?
#
class SparseSTDPConnectionMatrix(ConnectionMatrix):
    def __init__(self, val):
        if not isinstance(val, scipy.sparse.lil_matrix):
            raise TypeError('val should be a scipy sparse lil_matrix')
        nnz = val.getnnz()
        alldata = numpy.zeros(nnz)
        alli = numpy.zeros(nnz, dtype=int)
        allj = numpy.zeros(nnz, dtype=int)
        rowind = numpy.zeros(len(val.rows)+1, dtype=int)
        rowdata = []
        rowj = []
        coli = []
        coldataindices = []
        i = 0
        for r, d, c in zip(val.rows, val.data, xrange(len(val.rows))):
            rowind[c] = i
            alldata[i:i+len(d)] = d
            allj[i:i+len(r)] = r
            alli[i:i+len(r)] = c
            rowdata.append(alldata[i:i+len(d)])
            rowj.append(allj[i:i+len(r)])
            i = i+len(r)
        rowind[len(val.rows)] = i
        counts = numpy.histogram(allj, numpy.arange(val.shape[1]+1, dtype=int), new=True)[0]
        curcdi = numpy.zeros(val.shape[1], dtype=int)
        for j in xrange(val.shape[1]):
            coldataindices.append(numpy.zeros(counts[j], dtype=int))
        for i, j in enumerate(allj):
            coldataindices[j][curcdi[j]] = i
            curcdi[j]+=1
        for j in xrange(val.shape[1]):
#            coldataindices.append((allj==j).nonzero()[0])
            coli.append(alli[coldataindices[j]])
        self.alldata = alldata
        self.rowdata = rowdata
        self.allj = allj
        self.alli = alli
        self.rowj = rowj
        self.coli = coli
        self.coldataindices = coldataindices
        self.rowind = rowind
        self.shape = val.shape
        self.rowret = numpy.zeros(val.shape[1])
        self.colret = numpy.zeros(val.shape[0])
        self.rowset = numpy.zeros(val.shape[1])
        self.colset = numpy.zeros(val.shape[0])
        self._useaccel = get_global_preference('useweave')
        self._cpp_compiler = get_global_preference('weavecompiler')
    def get_row(self, i):
        self.rowret[:]=0
        self.rowret[self.rowj[i]] = self.rowdata[i]
        return self.rowret
    def set_row(self, i, val):
        self.rowset[:] = val
        self.rowdata[i][:] = self.rowset[self.rowj[i]]
    def get_col(self, j):
        self.colret[:]=0
        self.colret[self.coli[j]] = self.alldata[self.coldataindices[j]]
        return self.colret
    def set_col(self, j, val):
        self.colset[:] = val
        self.alldata[self.coldataindices[j]] = self.colset[self.coli[j]]
    def get_pointer(self, i, j):
        I, = (self.coli[j]==i).nonzero()
        if len(I)==1:
            return self.coldataindices[j][I[0]]
        return None
    # we support the following indexing schemes:
    # - s[:]
    # - s[i,:]
    # - s[:,i]
    # - s[i,j]
    def __getitem__(self, item):
        if isinstance(item,tuple) and isinstance(item[0],int) and item[1]==colon_slice:
            return self.get_row(item[0])
        if isinstance(item,slice):
            if item==colon_slice:
                return self.alldata
            else:
                raise ValueError(str(item)+' not supported.')
        if isinstance(item,int):
            return self.get_row(item)
        if isinstance(item,tuple):
            if len(item)!=2:
                raise TypeError('Only 2D indexing supported.')
            item_i, item_j = item
            if isinstance(item_i, int) and isinstance(item_j, slice):
                if item_j==colon_slice:
                    return self.get_row(item_i)
                raise ValueError('Only ":" indexing supported.')
            if isinstance(item_i, slice) and isinstance(item_j, int):
                if item_i==colon_slice:
                    return self.get_col(item_j)
                raise ValueError('Only ":" indexing supported.')
            if isinstance(item_i, int) and isinstance(item_j, int):
                pointer = self.get_pointer(item_i, item_j)
                if pointer is None:
                    return 0.0
                return self.alldata[pointer]
            raise TypeError('Only (i,:), (:,j) and (i,j) indexing supported.')
        raise TypeError('Can only get items of type slice or tuple')
    def __setitem__(self, item, val):
        if isinstance(item,tuple) and isinstance(item[0],int) and item[1]==colon_slice:
            return self.set_row(item[0],val)
        if isinstance(item,slice):
            if item==colon_slice:
                self.alldata[:] = val
                return
            else:
                raise ValueError(str(item)+' not supported.')
        if isinstance(item,int):
            self.set_row(item, val)
            return
        if isinstance(item,tuple):
            if len(item)!=2:
                raise TypeError('Only 2D indexing supported.')
            item_i, item_j = item
            if isinstance(item_i, int) and isinstance(item_j, slice):
                if item_j==colon_slice:
                    self.set_row(item_i, val)
                    return
                raise ValueError('Only ":" indexing supported.')
            if isinstance(item_i, slice) and isinstance(item_j, int):
                if item_i==colon_slice:
                    self.set_col(item_j, val)
                    return
                raise ValueError('Only ":" indexing supported.')
            if isinstance(item_i, int) and isinstance(item_j, int):
                pointer = self.get_pointer(item_i, item_j)
                if pointer is None:
                    return
                self.alldata[pointer] = val
                return
            raise TypeError('Only (i,:), (:,j) and (i,j) indexing supported.')
        raise TypeError('Can only get items of type slice or tuple')
    def __str__(self):
        s = ''
        for i in range(self.shape[0]):
            s += str(self.get_row(i)) + '\n'
        for j in range(self.shape[1]):
            s += str(self.get_col(j)) + '\n'
        return s
    def add_rows(self,spikes,X):
        if not len(spikes): return
        if self._useaccel:
            if not isinstance(spikes, numpy.ndarray):
                spikes = array(spikes, dtype=int)
            alldata = self.alldata
            allj = self.allj
            rowind = self.rowind
            nspikes = len(spikes)
            code =  """
                    for(int i=0;i<nspikes;i++)
                    {
                        int j = spikes(i);
                        int ri_start = rowind(j);
                        int ri_end = rowind(j+1);
                        for(int ri=ri_start;ri<ri_end;ri++)
                            X(allj(ri)) += alldata(ri);
                    }
                    """
            weave.inline(code,['X','alldata','allj','rowind','spikes','nspikes'],
                         compiler=self._cpp_compiler,
                         type_converters=weave.converters.blitz,
                         extra_compile_args=['-O3'])
        else:
            for i in spikes:
                X[self.rowj[i]] += self.rowdata[i]

if __name__=='__main__':
    a = scipy.sparse.lil_matrix((3,4))
    a[0,0] = 1
    a[0,1] = 2
    a[1,2] = 3
    a[2,1] = 4
    #print a.todense()
    s = SparseSTDPConnectionMatrix(a)
    print s
    print s[:]
    print s[0,:]
    print s[:,1]
    print s[1,2]
    print s[1,3]
    print
    s[:] = 5
    print s
    s[0,:] = [ 6.1, 6.2, 6.3, 6.4 ]
    print s
    s[:,1] = [ 7.1, 7.2, 7.3 ]
    print s
    s[1,2] = 8
    print s
    s[1,3] = 9
    print s
#    s.set_row(0, 5)
#    s.set_row(0, [6,0,7,0])
#    s.set_col(1, 8)
#    s.set_col(1, numpy.array([9,10,11]))
