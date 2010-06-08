# ----------------------------------------------------------------------------------
# Copyright ENS, INRIA, CNRS
# Contributors: Romain Brette (brette@di.ens.fr) and Dan Goodman (goodman@di.ens.fr)
# 
# Brian is a computer program whose purpose is to simulate models
# of biological neural networks.
# 
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software.  You can  use, 
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info". 
# 
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability. 
# 
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or 
# data to be ensured and,  more generally, to use and operate it in the 
# same conditions as regards security. 
# 
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# ----------------------------------------------------------------------------------
# 
'''
Circular arrays

Ideas for speed improvements: use put, putmask and take with mode='wrap' and out=...
'''
from numpy import *
from scipy import weave
import bisect
import os
import warnings
from ..globalprefs import get_global_preference

__all__ = ['CircularVector', 'SpikeContainer']


class CircularVector(object):
    '''
    A vector with circular structure.
    Variables:
    * X = the data (array of size n)
    * cursor = current position in the array (where the 0 index is)
    '''
    def __init__(self, n, dtype=float, useweave=False, compiler=None): # pylint: disable-msg=W0621
        '''
        n is the size of the vector.
        '''
        self.X = zeros(n, dtype=dtype)
        self.dtype = dtype
        self.cursor = 0
        self.n = n
        self._useweave = useweave
        if useweave:
            self._optimisedreturnarray = zeros(n, dtype=dtype)
            self._cpp_compiler = compiler
            self._extra_compile_args = ['-O3']
            if self._cpp_compiler == 'gcc':
                self._extra_compile_args += get_global_preference('gcc_options') # ['-march=native', '-ffast-math']
        else:
            self._cpp_compiler = ''

    def reinit(self):
        self.X[:] = zeros(self.n, self.dtype)
        self.cursor = 0

    def advance(self, k):
        self.cursor = (self.cursor + k) % self.n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        '''
        V[i]
        '''
        return self.X[(self.cursor + i) % self.n]

    def __setitem__(self, i, x):
        '''
        V[i]=x
        '''
        self.X[(self.cursor + i) % self.n] = x

    def __getslice__(self, i, j):
        n = self.n
        i0 = (self.cursor + i) % n # pylint: disable-msg=W0621
        j0 = (self.cursor + j) % n # pylint: disable-msg=W0621
        if j0 >= i0:
            return self.X[i0:j0]
        else:
            #return self.X[range(i0,n)+range(0,j0)]
            return concatenate((self.X[i0:], self.X[0:j0])) # this version is MUCH faster

    def get_conditional(self, i, j, min, max, offset=0):
        """
        Returns only those vectors with values between min and max
        
        This rather specialised usage of a circular vector is for the
        benefit of the SpikeContainer class get_spikes method, which is
        in turn used by the NeuronGroup.get_spikes method.
        
        It returns v-offset for those elements v in self[i:j] such that min<v<max. 
        """
        if self._useweave:
            n = self.n
            i0 = int((self.cursor + i) % n) # pylint: disable-msg=W0612,W0621
            j0 = int((self.cursor + j) % n) # pylint: disable-msg=W0612
            X = self.X # pylint: disable-msg=W0612
            ret = self._optimisedreturnarray
            code = """
                    int numgot = 0;
                    for(int k=i0;k!=j0;k=(k+1)%n)
                    {
                        int Xk = X(k);
                        if(Xk>=min && Xk<max)
                            ret(numgot++)=Xk-offset;
                    }
                    return_val = numgot;
                    """
            numgot = weave.inline(code, ['n', 'i0', 'j0', 'X', 'ret', 'offset', 'min', 'max'],
                                     compiler=self._cpp_compiler,
                                     type_converters=weave.converters.blitz,
                                     extra_compile_args=self._extra_compile_args)
            return ret[0:numgot]
        else:
            spikes = self[i:j]
            spikes = spikes[bisect.bisect_left(spikes, min):bisect.bisect_left(spikes, max)]
            if offset: spikes = spikes - offset
            return spikes

    def __setslice__(self, i, j, W):
        # NB: S[4:1] does not give a circular slice but only []
        # Should we change that behaviour?
        # TODO: speed improvements
        if j > i:
            n = self.n
            i0 = (self.cursor + i) % n # pylint: disable-msg=W0621
            j0 = (self.cursor + j) % n
            if j0 > i0:
                self.X[i0:j0] = W
            elif isinstance(W, ndarray):
                self.X[i0:] = W[0:n - i0]
                self.X[0:j0] = W[n - i0:n - i0 + j0]

    def __repr__(self):
        return repr(hstack((self[0:self.n - 1], self[self.n - 1:self.n])))

    def __print__(self):
        return (hstack((self[0:self.n - 1], self[self.n - 1:self.n]))).__print__()


class SpikeContainer(object):
    '''
    An object that stores previous spikes.
    S[0] is an array of the last spikes (neuron indexes).
    S[1] is an array with the spikes at time t-dt, etc.
    S[0:50] contains all spikes in last 50 bins.
    '''
    def __init__(self, m, useweave=False, compiler=None):
        '''
        n = maximum number of spikes stored
        m = maximum number of bins stored
        '''
        if m < 2: m = 2
        self.S = CircularVector(2, dtype=int, useweave=useweave, compiler=compiler)
        self.ind = CircularVector(m + 1, dtype=int, useweave=useweave, compiler=compiler) # indexes of bins
        self.remaining_space = 1
        self._useweave = useweave

    def reinit(self):
        self.S.reinit()
        self.ind.reinit()

    def push(self, spikes):
        '''
        Stores spikes in the array at time dt.
        '''
        ns = len(spikes)
        self.remaining_space += (self.ind[2] - self.ind[1]) % self.S.n
        while ns >= self.remaining_space:
            # double size of array
            S = self.S
            newS = CircularVector(2 * S.n, dtype=int, useweave=S._useweave, compiler=S._cpp_compiler)
            newS.X[:S.n - S.cursor] = S.X[S.cursor:]
            newS.X[S.n - S.cursor:S.n] = S.X[:S.cursor]
            newS.cursor = S.n
            self.S = newS
            self.ind.X = (self.ind.X - S.cursor) % S.n
            self.ind.X[self.ind.X == 0] = S.n
            self.remaining_space += S.n
        self.S[0:ns] = spikes
        self.S.advance(ns)
        self.ind.advance(1)
        self.ind[0] = self.S.cursor
        self.remaining_space -= ns

    def lastspikes(self):
        '''
        Returns S[0].
        '''
        return self.S[self.ind[-1] - self.S.cursor:self.S.n]

    def __getitem__(self, i):
        '''
        S[i]: returns the spikes at time t-i*dt.
        '''
        # NB: this could be optimized
        return self.S[self.ind[-i - 1] - self.S.cursor:self.ind[-i] - self.S.cursor + self.S.n]

    # optimised version of the above, but the speed improvement is not very much, might be
    # better to just wait and write a fully C/C++ version of the whole library
    def get_spikes(self, delay, origin, N):
        """
        Returns those spikes in self[delay] between origin and origin+N
        """
        return self.S.get_conditional(self.ind[-delay - 1] - self.S.cursor, \
                                     self.ind[-delay] - self.S.cursor + self.S.n, \
                                     origin, origin + N, origin)

    def __getslice__(self, i, j):
        return self.S[self.ind[-j] - self.S.cursor:self.ind[-i] - self.S.cursor + self.S.n]

    def __repr__(self):
        return "Spike container."

    def __print__(self):
        return self.__repr__()

try:
    import ccircular.ccircular as _ccircular
    class SpikeContainer(_ccircular.SpikeContainer):
        def __init__(self, m, useweave=False, compiler=None):
            _ccircular.SpikeContainer.__init__(self, m)
    #warnings.warn('Using C++ SpikeContainer')
except ImportError:
    pass

# I am not sure that class below is useful!
class ModInt(object):
    '''
    A number in Z/nZ, i.e., modulo n.
    Variables:
    * x = number modulo n
    * n
    Implemented: additions and subtraction.
    N.B.: not a very useful class.
    '''
    def __init__(self, x, n):
        self.x = x
        self.n = n

    def __add__(self, m):
        if isinstance(m, ModInt):
            assert self.n == m.n
            return ModInt((self.x + m.x) % self.n, self.n)
        else:
            return ModInt((self.x + m) % self.n, self.n)

    def __radd__(self, m):
        if isinstance(m, ModInt):
            assert self.n == m.n
            return ModInt((self.x + m.x) % self.n, self.n)
        else:
            return ModInt((self.x + m) % self.n, self.n)

    def __sub__(self, m):
        if isinstance(m, ModInt):
            assert self.n == m.n
            return ModInt((self.x - m.x) % self.n, self.n)
        else:
            return ModInt((self.x - m) % self.n, self.n)

    def __repr__(self):
        return str(self.x) + ' [' + str(self.n) + ']'

    def __print__(self):
        return self.__repr__()
