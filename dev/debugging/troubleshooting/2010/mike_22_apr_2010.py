from brian import *
from brian.library.random_processes import *
import multiprocessing

# Parameters
C=281*pF
gL=30*nS
taum=C/gL
EL=-70.6*mV
VT=-50.4*mV
DeltaT=2*mV
Vcut=VT+5*DeltaT
taue=2.728*ms
taui=10.49*ms
Ee=0*mV
Ei=-75*mV

eGain=1
iGain=2
# Pick an electrophysiological behaviour
tauw, a, b, Vr=144*ms, 4*nS, 0.0805*nA,-70.6*mV # Regular spiking (as inthe paper)
#tauw,a,b,Vr=20*ms,4*nS,0.5*nA,VT+5*mV # Bursting
#tauw,a,b,Vr=144*ms,2*C/(144*ms),0*nA,-70.6*mV # Fast spiking

eqs="""
dvm/dt=(gL*(EL-vm)+gL*DeltaT*exp((vm-VT)/DeltaT)+I-w+eGain*ge*(Ee-vm)+iGain*gi*(Ei-vm))/C  : volt
dw/dt=(a*(vm-EL)-w)/tauw : amp
I : amp
"""
eqs+=OrnsteinUhlenbeck('ge', mu=2.*gL/3, sigma=2.*gL/3, tau=taue)
eqs+=OrnsteinUhlenbeck('gi', mu=4.*gL/3, sigma=4.*gL/3, tau=taui)

"""Trace is a non-dimensionalized array of voltages a StateMonitor
recorded. Spikes is the accompanying SpikeMonitor passed to ease
finding spikes in trace. Cutoff is the voltage at which to truncate
each spike."""
def formatTraces(trace, spikes, cutoff, dt):
    answer=trace
    for _, t in spikes.spikes:answer[int(t/defaultclock.dt)]=20*mV
    return answer

# This is the function that we want to compute for various different parameters
def singleNeuronTrial(duration):
    # These two lines reset the clock to 0 and clear any remaining data so that
    # memory use doesn't build up over multiple runs.
    reinit_default_clock()
    clear(True)
    neuron=NeuronGroup(1, model=eqs, threshold=Vcut, reset="vm=Vr;w+=b", freeze=True)
    neuron.vm=EL
    trace=StateMonitor(neuron, 'vm', record=True)
    cutoff=20*mV
    spikes=SpikeMonitor(neuron)
    excitation=StateMonitor(neuron, 'ge', record=True)
    inhibition=StateMonitor(neuron, 'gi', record=True)
    run(duration*ms)
    formattedTrace=format(trace[0], spikes, cutoff, defaultclock.dt)
    return formattedTrace, trace.times/ms, excitation, inhibition

def visualize(results):

    #Compute Averages
    traces=mean(results[0])
    times=results[1]
    excitations=mean(results[2])
    exTimes=results[2].times
    inhibitions=mean(results[3])
    inTimes=results[3].times

    figure()
    subplot(211)
    title('Average AdExp Basal Activity over 100 2s Trial')
    ylabel('Voltage (mV)')
    plot(times/ms, traces/mV)
    subplot(212)
    plot(exTimes/ms, (excitations*(Ee-traces))/nA)
    plot(Intimes/ms, (inhibitions*(Ei-traces))/nA, 'r')
    title('Background Activity')
    ylabel('Current (nA)')
    xlabel('Time (ms)')
    legend(('Excitation', 'Inihibition'))

if __name__=='__main__':
    pool=multiprocessing.Pool() # uses num_cpu processes by default
    runNumber=10
    duration=20
    args=[duration]*runNumber
    #results = pool.map(singleNeuronTrial, args)
    results=map(singleNeuronTrial, args)
    visualize(results)
    show()
