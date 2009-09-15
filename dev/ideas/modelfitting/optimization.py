from scipy import * 
from time import *
import pickle

def optimize(X0, fun, iterations, pso_params, min_values = None, max_values = None, group_size = None):
    """
    Computes the argument of fun which maximizes it using the Particle Swarm Optimization algorithm.
    
    INPUTS
    X0 is a N*M matrix
    N is the space dimension
    M is the number of particles
    If constraints is set, it is a N-long array such that X >= constraints.
    fun(x0) is a 1*M row vector
    If group_number is set, it means that particles are grouped within groups of size M/group_number
    The objective function is assumed to make a different computation for each group.
    The PSO algorithm then maximizes f in each group independently. It returns a matrix
    of size N*group_number : each column is the result of the optimization for the corresponding group.
    
    OUTPUTS
    pso(x0, fun) returns argmax fun(x)
    pso_state_filename is the filename of the state
    of the optimization algorithm at a certain iteration.
    If set, the optimization is resumed after that iteration.
    """

    (N,M) = X0.shape
    if group_size is None:
        group_size = M
    group_number = M/group_size
    
    if (min_values is None):
        min_values = -inf*ones((N,M))
    
    if (max_values is None):
        max_values = inf*ones((N,M))
    
    fitness_X = -inf * ones((1,M))
    fitness_lbest = fitness_X
    fitness_gbest = -inf*ones(group_number)
    
    omega = pso_params[0]
    c1 = pso_params[1]
    c2 = pso_params[2]
    
    X0 = maximum(X0, min_values)
    X0 = minimum(X0, max_values)
    X = X0
    V = zeros((N,M))
    
    X_lbest = X
    X_gbest = X[:,arange(0,M,group_size)] # N*group_number matrix
    
    time0 = clock()
    for k in range(iterations):
        print 'Step %d/%d' % (k+1 , iterations)
        R1 = rand(N,M)
        R2 = rand(N,M)
        X_gbest2 = zeros((N, M))
        for j in range(group_number):
            X_gbest2[:,j*group_size:(j+1)*group_size] = tile(X_gbest[:,j].reshape(N,1), (1, group_size))
        V = omega*V + c1*R1*(X_lbest-X) + c2*R2*(X_gbest2-X)
        X = X + V
        
        X = maximum(X, min_values)
        X = minimum(X, max_values)
        
        time1 = clock()
        fitness_X = fun(X)
        time2 = clock()
        
        # Local update
        indices_lbest = nonzero(fitness_X > fitness_lbest)[1]
        if (len(indices_lbest)>0):
            X_lbest[:,indices_lbest] = X[:,indices_lbest]
            fitness_lbest[:,indices_lbest] = fitness_X[:,indices_lbest]
        
        # Global update
        max_fitness_X = array([max(fitness_X[j*group_size:(j+1)*group_size]) for j in range(group_number)])
        for j in nonzero(max_fitness_X > fitness_gbest)[0]: # groups for which a global best has been reached at this iteration
            sub_fitness_X = fitness_X[j*group_size:(j+1)*group_size]
            index_gbest = nonzero(sub_fitness_X == max_fitness_X[j])[0]
            if not(isscalar(index_gbest)):
                index_gbest = index_gbest[0]
            X_gbest[:,j] = X[:,j*group_size+index_gbest]
            fitness_gbest[j] = max_fitness_X[j]
       
        print 'Evaluation time :', '%.2f s' % (time2-time1)
        print 'Total elapsed time :', '%.2f s' % (clock()-time0)
#        print 'Mean best value :', '%.5f' % fitness_gbest.mean()
        for j in range(group_number):
            print 'Best value %d :' % (j+1), '%.5f' % fitness_gbest[j]
        print
        
    return (X_gbest, fitness_gbest, clock()-time0)


if __name__ == "__main__":
    group_number = 5
    group_size = 10
    def fun(x):
        y = exp(-.7*sum(x**2, axis=0))
        return y
    result = optimize(X0 = 3*rand(5,group_size*group_number), fun = fun, iterations = 500, pso_params = [.8, 1.8, 1.8], group_size = group_size)

#    print '-------------------------------'
#    print '-------------------------------'
#    print 'Final value : %.10f' % result
    