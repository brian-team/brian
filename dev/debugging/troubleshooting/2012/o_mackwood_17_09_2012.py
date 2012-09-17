import brian_no_units
from brian.globalprefs import set_global_preferences

set_global_preferences(useweave=True,
                       usecodegen=True,
                       usecodegenweave=True,
                       usecodegenstateupdate = True,
                       usenewpropagate = True,
                       usecstdp = True,
                       gcc_options = ['-O3', '-ffast-math','-march=core2']
                       )

def run_stdp(NE,NI,v_init,C_e,C_ii,C_ie,mon_bin,dt):
    from brian.neurongroup import NeuronGroup
    from brian.monitor import PopulationRateMonitor
    from brian.stdunits import mV, ms, nS, pF, pA, Hz
    from brian.units import second
    from brian.equations import Equations
    from brian.network import Network
    from brian.connections import Connection
    from brian.stdp import STDP
    from brian.clock import Clock

    runtime = 10*second
   
    eta = 1e-2          # Learning rate
    tau_stdp = 20*ms    # STDP time constant
    alpha = 3*Hz*tau_stdp*2  # Target rate parameter
    gmax = 100               # Maximum inhibitory weight

    eqs_neurons='''
    dv/dt=(-gl*(v-el)-(g_ampa*w*v+g_gaba*(v-er)*w)+bgcurrent)/memc : volt
    dg_ampa/dt = -g_ampa/tau_ampa : 1
    dg_gaba/dt = -g_gaba/tau_gaba : 1
    '''
    namespace = {'tau_ampa':5.0*ms,'tau_gaba':10.0*ms,
                 'bgcurrent':200*pA,'memc':200.0*pF,
                 'el':-60*mV,'w':1.*nS,'gl':10.0*nS,'er':-80*mV}
    eqs_neurons = Equations(eqs_neurons, ** namespace)

    clock = Clock(dt)
    neurons=NeuronGroup(NE+NI,model=eqs_neurons,clock=clock,
                        threshold=-50.*mV,reset=-60*mV,refractory=5*ms)
    neurons.v = v_init
    Pe=neurons.subgroup(NE)
    Pi=neurons.subgroup(NI)
    rme = PopulationRateMonitor(Pe,mon_bin)
    rmi = PopulationRateMonitor(Pi,mon_bin)
   
    con_e = Connection(Pe,neurons,'g_ampa')
    con_ie = Connection(Pi,Pe,'g_gaba')
    con_ii = Connection(Pi,Pi,'g_gaba')
    con_e.connect_from_sparse(C_e, column_access=True)
    con_ie.connect_from_sparse(C_ie, column_access=True)
    con_ii.connect_from_sparse(C_ii, column_access=True)

    eqs_istdp = '''
    dA_pre/dt=-A_pre/tau_stdp : 1
    dA_post/dt=-A_post/tau_stdp : 1
    '''
    stdp_params = {'tau_stdp':tau_stdp, 'eta':eta, 'alpha':alpha}
    eqs_istdp = Equations(eqs_istdp, **stdp_params)
    stdp_ie = STDP(con_ie, eqs=eqs_istdp,
                   pre='''A_pre+=1.
                          w+=(A_post-alpha)*eta''',
                   post='''A_post+=1.
                           w+=A_pre*eta''',
                   wmax=gmax)
   
    net = Network(neurons, con_e, con_ie, con_ii, stdp_ie, rme, rmi)
    net.run(runtime,report='text')
    return (rme.times,rme.rate), (rmi.times,rmi.rate)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from brian.network import clear
    from brian.connections.construction import random_matrix
    from brian.stdunits import mV,ms
    import numpy as np

    NE = 8000          # Number of excitatory cells
    NI = NE/4          # Number of inhibitory cells
    epsilon = 0.15      # Sparseness of synaptic connections
   
    mon_bin = 1*ms
    dt = 0.1*ms

    C_e  = random_matrix(NE, NE+NI,epsilon, value=0.3)
    C_ii = random_matrix(NI, NI,   epsilon, value=3.0)
    C_ie = random_matrix(NI, NE,   epsilon, value=2.0)
    v_init = -80*np.random.rand(NE+NI)*mV

    rme1, rmi1 = run_stdp(NE,NI,v_init,C_e,C_ii,C_ie,mon_bin,dt)
    clear()
    rme2, rmi2 = run_stdp(NE,NI,v_init,C_e,C_ii,C_ie,mon_bin,dt)

    filter_width = 200*ms
    width_dt = int(filter_width / mon_bin) # width in number of bins
    window = np.exp(-np.arange(-2 * width_dt, 2 * width_dt + 1) ** 2 * 1. / (2 * (width_dt) ** 2))

    fig = plt.figure()
    ax_inh = fig.add_subplot(2,1,1)
    ax_exc = fig.add_subplot(2,1,2)
    for rme,rmi,run_str in [(rme1,rmi1,'First'),(rme2,rmi2,'Second')]:
        ax_exc.plot(rme[0],np.convolve(rme[1], window * 1. / np.sum(window), mode='same'),
                    label='Exc - '+run_str+' run')
        ax_inh.plot(rmi[0],np.convolve(rmi[1], window * 1. / np.sum(window), mode='same'),
                    label='Inh - '+run_str+' run')
    ax_exc.legend()
    ax_inh.legend()
    plt.show()
