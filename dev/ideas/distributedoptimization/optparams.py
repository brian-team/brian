from numpy import *
from numpy.random import rand, randn

__all__ = ['OptParams']

class OptParams(object):
    """Internal class used to manipulate model fitting parameters.
    It basically handles conversion between parameter dictionaries and arrays.
    
    Initialized with arguments:
    
    ``**params``
        Parameters list ``param_name=[bound_min, min_ max, bound_max]``
        
    **Methods**
    
    .. method:: get_initial_param_values(N)
    
        Returns initial parameter values sampled uniformly within the parameter 
        interval given in the constructor of the class. ``N`` is the number of neurons.
        The result is a dictionary ``{param_name=values}`` where values is a vector of values.
    
    .. method:: set_constraints()
    
        Returns the constraints for each parameter. The result is (min_values, max_values)
        where each variable is a vector containing the minimum and maximum values for each parameter.
    
    .. method:: get_param_values(X)
    
        Converts an array containing parameter values into a dictionary.
    
    .. method:: get_param_matrix(param_values)
    
        Converts a dictionary containing parameter values into an array.
    """
    def __init__(self, **params):
        self.params = params
        self.param_names = sort(params.keys())
        self.param_count = len(params)
        
    def get_initial_param_values(self, N):
        """
        Samples random initial param values around default values
        """
        random_param_values = {}
        for key, value in self.params.iteritems():
            if isscalar(value):
                value = [value]
#            if len(value) == 1:
#                # One default value, no boundary counditions on parameters
#                random_param_values[key] = value[0]*(1+.5*randn(N))
            elif len(value) == 2:
                # One default interval, no boundary counditions on parameters
                random_param_values[key] = value[0] + (value[1]-value[0])*rand(N)
#            elif len(value) == 3:
#                # One default value, value = [min, init, max]
#                random_param_values[key] = value[1]*(1+.5*randn(N))
            elif len(value) == 4:
                # One default interval, value = [min, init_min, init_max, max]
                random_param_values[key] = value[1] + (value[2]-value[1])*rand(N)
        return random_param_values

    def set_constraints(self):
        """
        Returns constraints of a given model
        returns min_values, max_values
        min_values is an array of length p where p is the number of parameters
        min_values[i] is the minimum value for parameter i
        """
        min_values = []
        max_values = []
        param_names = self.param_names
        p = self.param_count
        for key in param_names:
            value = self.params[key]
            # No boundary conditions if only two values are given
            if len(value) == 2:
                min_values.append(-inf)
                max_values.append(inf)
            else:
                min_values.append(value[0])
                max_values.append(value[-1])
        
        min_values = array(min_values)
        max_values = array(max_values)
        
        return min_values, max_values

    def get_param_values(self, X):
        """
        Converts a matrix containing param values into a dictionary
        """
        param_values = {}
        if X.ndim == 1:
            X = X.reshape((-1,1))
        for i in range(len(self.param_names)):
            param_values[self.param_names[i]] = X[i,:]
        return param_values
    
    def get_param_matrix(self, param_values):
        """
        Converts a dictionary containing param values into a matrix
        """
        p = self.param_count
        # Number of parameter values (number of particles)
        n = len(param_values[self.param_names[0]])
        X = zeros((p, n))
        for i in range(p):
            X[i,:] = param_values[self.param_names[i]]
        return X
    