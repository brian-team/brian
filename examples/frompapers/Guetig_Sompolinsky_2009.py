#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Implementation of the basic model (no speech recognition, no learning) 
described in:
Gutig and Sompolinsky (2009): "Time-Warp-Invariant Neuronal Processing"
PLoS Biology, Vol. 7 (7), e1000141
'''
from brian import *

class TimeWarpModel(object):
    '''
    A simple neuron model for testing the "time-warp invariance" with 
    conductance based or current based synapses. The neuron receives balanced
    excitatory and inhibitory input from a random spike train. The same spike
    train can be fed into the model with different time warps.
    ''' 
    def __init__(self, conductance_based=True):
        '''
        Create a new model with conductance based or current based synapses
        '''
        # Model parameters
        E_e = 5
        E_i = -1
        E_L = 0
        g_L = 1/(100.0*msecond)
        tau_syn = 1*ms
        N_ex = 250
        N_inh = 250
        self.N = N_ex + N_inh
                
        # Equations
        if conductance_based:
            eqs = '''
                dV/dt = -(V - E_L) * g_L - I_syn  : 1
                I_syn = I_ge + I_gi  : second**-1
                I_ge = (V - E_e) * g_e : second**-1
                I_gi = (V - E_i) * g_i : second**-1
                dg_e/dt = -g_e/tau_syn : second**-1
                dg_i/dt = -g_i/tau_syn : second**-1
                '''
        else:
            eqs = '''
                dV/dt = -(V - E_L) * g_L - I_syn  : 1
                I_syn = -5 * g_e + g_i : second**-1
                dg_e/dt = -g_e/tau_syn : second**-1
                dg_i/dt = -g_i/tau_syn : second**-1
                '''        
        
        # for simpler voltage traces: no spiking
        neuron = NeuronGroup(1, model=eqs, threshold=None)
            
        # every input neuron fires once in a random interval
        self.unwarped_spiketimes = [(i, t * 250 * ms) for i, t in 
                                    zip(range(0, self.N), rand(self.N))]
        
        # final spiketimes will be set in the run function
        self.input = SpikeGeneratorGroup(self.N, [])
                
        e_input = self.input.subgroup(N_ex)
        i_input = self.input.subgroup(N_inh)        
        e_conn = Connection(e_input, neuron, 'g_e',
                                  weight=6 / (N_ex * tau_syn))
        i_conn = Connection(i_input, neuron, 'g_i',
                                  weight=5 * 6 / (N_ex * tau_syn)) 
        
        # record membrane potential
        self.monitor = StateMonitor(neuron, varname='V', record=True)

        # putting everything together
        self.net = Network(neuron, self.input, e_conn, i_conn, self.monitor)

    def run(self, beta=1.0):
        ''' 
        Run the network with the original spike train warped by a certain factor
        beta. Beta > 1 corresponds to an extended and beta < 1 to a shrinked
        input spike train.
        '''
        self.net.reinit()
        
        #warp spike train in time        
        self.input.spiketimes = [(i, beta*t) 
                                 for i, t in self.unwarped_spiketimes]        
        self.net.run(beta * 250*ms)
        
        #Return the voltage trace
        return (self.monitor.times, self.monitor[0])


if __name__ == '__main__':
    cond_model = TimeWarpModel(True)
    curr_model = TimeWarpModel(False)
    N = cond_model.N    
    
    # #########################################################################
    # Reproduce Fig. 2 from Gütig and Sompolinsky (2009)
    # #########################################################################
    beta = 2.0
    times1, v1 = cond_model.run(beta=1.0)
    times2, v2 = cond_model.run(beta=beta)
    maxtime = 250 * beta
    subplot(4, 1, 1)
    (neurons, times) = zip(*cond_model.unwarped_spiketimes)
    plot(array(times) / ms, neurons, 'g.')
    axis([0, maxtime, 0, N])
    xticks([])
    yticks([])
    title('Time-warp-invariant voltage traces (conductance-based)')
    
    subplot(4, 1, 2)
    plot(times1 / ms, v1, 'g')
    axis([0, maxtime, -1.5, 1.5])
    xticks([])
    yticks([])
    
    subplot(4, 1, 3)
    plot(array(times) * beta / ms, neurons, 'b.')
    axis([0, maxtime, 0, N])
    xticks([])
    yticks([1, 500])
    
    subplot(4, 1, 4)
    plot(times2 / ms, v2, 'b')
    plot(times1 / ms * beta, v1, 'g')
    axis([0, maxtime, -1.5, 1.5])
    xlabel('Time (ms)')
    xticks([0, 250, 500])
    yticks([-1, 1])
    show()
    # #########################################################################
    # Reproduce Fig. 3(C) from Gütig and Sompolinsky (2009), but for random
    # spike trains and not in a speech recognition task
    # #########################################################################

    betas = arange(0.2, 3.1, 0.1)
    #betas = array([1.0, 2.0])
    cond_results = []
    curr_results = []
    for beta in betas:
        print 'Testing warp factor %.1f' % beta
        cond_results.append(cond_model.run(beta))
        curr_results.append(curr_model.run(beta))
    
    figure()
    colors = mpl.cm.gist_earth((betas - betas[0]) / (betas[-1] - betas[0]))
    lookup = dict(zip(betas, colors))
    for beta, cond_result, curr_result in zip(betas, cond_results, 
                                              curr_results):
        times_cond, v_cond = cond_result
        times_curr, v_curr = curr_result 
        subplot(1,2,1)
        plot(times_cond / ms / beta, v_cond, color=lookup[beta])
        axis([0, 250, -1.5, 1.5])
        subplot(1,2,2)
        plot(times_curr / ms / beta, v_curr, color=lookup[beta])
        axis([0, 250, -1.5, 1.5])
    subplot(1,2,1)
    title('conductance based')
    subplot(1,2,2)  
    title('current based')
    show()