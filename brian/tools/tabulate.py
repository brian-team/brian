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
Tabulation of numerical functions.
'''

__all__ = ['Tabulate', 'TabulateInterp']

from brian.units import get_unit, Quantity, is_dimensionless
from brian.unitsafefunctions import array, arange, zeros
from numpy import NaN


class Tabulate(object):
    '''
    An object to tabulate a numerical function.
    
    Sample use::
    
      g=Tabulate(f,0.,1.,1000)
      y=g(.5)
      v=g([.1,.3])
      v=g(array([.1,.3]))
      
    Arguments of g must lie in [xmin,xmax).
    An IndexError is raised is arguments are above xmax, but
    not always when they are below xmin (it can give weird results).
    '''
    def __init__(self, f, xmin, xmax, n):
        self.xmin = xmin
        self.xmax = xmax
        self.dx = (xmax - xmin) / float(n)
        self.invdx = 1 / self.dx
        self.unit = get_unit(f(xmin))
        # Tabulation at midpoints
        x = xmin + (.5 + arange(n)) * self.dx
        try:
            self.f = f(x)
        except:
            # If it fails we try passing the values one by one
            self.f = zeros(n) * f(xmin) # for the unit
            for i in xrange(n):
                self.f[i] = f(x[i])

    def __call__(self, x):
        try: # possible problem if x is an array and an array is wanted
            return self.f[array((array(x) - self.xmin) * self.invdx, dtype=int)]
        except IndexError: # out of bounds
            return NaN * self.unit

    def __repr__(self):
        return 'Tabulated function with ' + str(len(self.f)) + ' points'


class TabulateInterp(object):
    '''
    An object to tabulate a numerical function with linear interpolation.
    
    Sample use::
    
      g=TabulateInterp(f,0.,1.,1000)
      y=g(.5)
      v=g([.1,.3])
      v=g(array([.1,.3]))
      
    Arguments of g must lie in [xmin,xmax).
    An IndexError is raised is arguments are above xmax, but
    not always when they are below xmin (it can give weird results).
    '''
    def __init__(self, f, xmin, xmax, n):
        self.xmin = xmin
        self.xmax = xmax
        self.dx = (xmax - xmin) / float(n - 1)
        self.invdx = 1 / self.dx
        # Not at midpoints here
        x = xmin + arange(n) * self.dx
        self.unit = get_unit(f(xmin))
        try:
            self.f = f(x)
        except:
            # If it fails we try passing the values one by one
            self.f = zeros(n) * f(xmin) # for the unit
            for i in xrange(n):
                self.f[i] = f(x[i])
        self.f = array(self.f)
        self.df = (self.f[range(1, n)] - self.f[range(n - 1)]) * float(self.invdx)

    def __call__(self, x): # the units of x is not checked
        y = array(x) - self.xmin
        ind = array(y * self.invdx, dtype=int)
        try:
            if is_dimensionless(x): # could be a problem if it is a Quantity with units=1
                return self.f[ind] + self.df[ind] * (y - array(ind) * self.dx)
            else:
                return array(self.f[ind] + self.df[ind] * (y - array(ind) * self.dx)) * self.unit
        except IndexError: # out of bounds
            return NaN * self.unit

    def __repr__(self):
        return 'Tabulated function with ' + str(len(self.f)) + ' points (interpolated)'
