import brian_no_units
from brian import *
import numpy
import brian
import brian.experimental.modelfitting as gpu
import fast_compensation as cpu
#brian.log_level_error()
debug_level()

def compensate_gpu(input, trace, p=1.0):
    
    equations = Equations('''
            dV0/dt=(R*Iinj-V0+Vr)/tau : 1
            Iinj=(V-V0)/Re : 1
            dV/dt=Re*(I-Iinj)/taue : 1
            Ve=V-V0 : 1
            I : 1
            R : 1
            Re : 1
            Vr : 1
            tau : second
            taue : second
        ''')
    
    params = dict(      R =  [1.0e3, 1.0e6, 1000.0e6, 1.0e12],
                        tau =  [.1*ms, 1*ms, 30*ms, 200*ms],
                        Vr = [-100.0e-3, -80e-3, -40e-3, -10.0e-3],
                        Re = [1.0e3, 1.0e6, 1000.0e6, 1.0e12],
                        taue = [.01*ms, .1*ms, 5*ms, 20*ms])
   
   
   
    tracecomp, traceV, traceVe, results = gpu.compensate(input,
                                                          trace,
                                                          gpu=1,
                                                          p=p,
                                                          equations=equations,
                                                          maxiter=10,
                                                          popsize=10,
                                                          slice_duration=1*second,
                                                          **params)
    
    return traceV, traceVe, results






if __name__ == '__main__':

    I = numpy.load("current1.npy")[:10000] * 25 # because current1[2010_12_09_0006] * 25
    Vraw = numpy.load("trace1.npy")[:10000]

    R = 600*Mohm
    tau = 10*ms
    Vr = -70*mV
    Re = 100*Mohm
    taue = 1*ms

    # CPU COMPENSATION
    #Vcomp, params = cpu.compensate(I, Vraw, dt=.1*ms,
    #                                p = 1.0,
    #                                durslice=10*second,
    #                                R=R, tau=tau, Vr=Vr, Re=Re, taue=taue)

    # GPU COMPENSATION
    traceV, traceVe, results = compensate_gpu(I, Vraw)
    Vcomp_gpu = Vraw - traceVe

    subplot(211)
    plot(I)

    subplot(212)
    plot(Vraw)
    plot(Vcomp_gpu)

    show()
