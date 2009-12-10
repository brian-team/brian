from brian.stdunits import ms
from numpy import *
from numpy.random import rand, randn

class FittingParameters(object):
    def __init__(self, includedelays = True, **params):
        self.params = params
        self.param_names = sort(params.keys())
        self.param_count = len(params)
        self.includedelays = includedelays
        
    def get_initial_param_values(self, N, includedelays = True):
        """
        Samples random initial param values around default values
        """
        random_param_values = {}
        for key, value in self.params.iteritems():
            if isscalar(value):
                value = [value]
            if len(value) == 1:
                # One default value, no boundary counditions on parameters
                random_param_values[key] = value[0]*(1+.5*randn(N))
            elif len(value) == 2:
                # One default interval, no boundary counditions on parameters
                random_param_values[key] = value[0] + (value[1]-value[0])*rand(N)
            elif len(value) == 3:
                # One default value, value = [min, init, max]
                random_param_values[key] = value[1]*(1+.5*randn(N))
            elif len(value) == 4:
                # One default interval, value = [min, init_min, init_max, max]
                random_param_values[key] = value[1] + (value[2]-value[1])*rand(N)
        if includedelays & self.includedelays:
            # Appends initial param values for spike delays, between +-5*ms
            random_param_values['delays'] = -5*ms + 10*ms*rand(N)
        return random_param_values

    def set_constraints(self, N, includedelays = True):
        """
        Returns constraints of a given model
        constraints is an array of length p where p is the number of parameters
        constraints[i] is the minimum value for parameter i
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
        
        if includedelays & self.includedelays:
            # Boundary conditions for delays parameter
            min_values.append(-5.0*ms)
            max_values.append(5.0*ms)
        
        min_values = array(min_values)
        max_values = array(max_values)
        
        if includedelays & self.includedelays:
            min_values = tile(min_values.reshape((p+1, 1)), (1, N))
            max_values = tile(max_values.reshape((p+1, 1)), (1, N))
        else:
            min_values = tile(min_values.reshape((p, 1)), (1, N))
            max_values = tile(max_values.reshape((p, 1)), (1, N))
            
        return min_values, max_values

    def get_param_values(self, X, includedelays = True):
        """
        Converts a matrix containing param values into a dictionary
        """
        param_values = {}
        for i in range(len(self.param_names)):
            param_values[self.param_names[i]] = X[i,:]
        if includedelays & self.includedelays:
            # Last row in X = delays
            param_values['delays'] = X[-1,:]
        return param_values
    
    def get_param_matrix(self, param_values, includedelays = True):
        """
        Converts a dictionary containing param values into a matrix
        """
        p = self.param_count
        # Number of parameter values (number of particles)
        n = len(param_values[self.param_names[0]])
        if includedelays & self.includedelays:
            # Last row in X = delays
            X = zeros((p+1, n))
            X[-1,:] = param_values['delays']
        else:
            X = zeros((p, n))
        for i in range(p):
            X[i,:] = param_values[self.param_names[i]]
        return X
    