from brian.utils.particle_swarm import *

# IF example, parameters = R and tau
# N is the number of neurons/particles
# X is a 2*N matrix :
#   X[0,:] contains the R parameters for all neurons
#   X[1,:] contains the tau parameters for all neurons
# fun(X) returns an array with the gamma factors of all neurons

def fun(X):
    R = X[0,:]
    tau = X[1,:]
    # TODO: set the parameters in the group
    # TODO: run the simulation
    # TODO: compute the gamma factor
    # TODO: return the gamma factor

# TODO: set X0

X, value, T = particle_swarm(X0, fun, iterations = 10, pso_params = [.9, 1.9, 1.9])

final_R = X[0,:]
final_tau = X[1,:]
# value is the final gamma factor

