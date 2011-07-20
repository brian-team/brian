from brian import *
from modelfitting import *
import time, os

dt = defaultclock.dt

def compensate(current, trace, popsize = 100, maxiter = 10,
               equations = None, reset = None, threshold = None,
                  slice_duration = 1 * second, overlap = 100*ms,
                  initial = None, dt = defaultclock.dt,
                  cpu = None, gpu = 1, record=['V','Ve'],
                  cut = None, cut_length = 0.0,
                  p = 0.5, best_params=None,
                  **params):
    
    trace0 = trace.copy()
    
    if initial is None: 
        initial = trace[0]
    
    if slice_duration is None:
        slices = 1
    else:
        slices = max(int(len(current)*dt/slice_duration), 1)
    current0 = current
    current, trace = slice_trace(current, trace, slices=slices, overlap=overlap, dt=dt)
    
    if cut is not None:
        cut, cut_indices = transform_spikes(popsize, current0, [(0, float(c)) for c in cut], 
                                            slices=slices, overlap=overlap, dt=dt)
        cut_steps = int32(cut_length/dt)
        cut = array(cut/dt, dtype=int32)
        cut_indices = array(cut_indices, dtype=int32)
    else:
        cut_indices = None
        cut_steps = 0
    
    initial_values = {'V0': initial, 'Ve': 0.}
    
    if equations is None:
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
    if threshold is None:
        threshold = "V>100000"
    if reset is None:
        reset = ""
    
    if len(params) == 0:
        params = dict(R =  [1.0e3, 1.0e6, 1000.0e6, 1.0e12],
                      Re = [1.0e3, 1.0e6, 1000.0e6, 1.0e12],
                      Vr = [-100.0e-3, -80e-3, -40e-3, -10.0e-3],
                      tau =  [.1*ms, 1*ms, 30*ms, 200*ms],
                      taue = [.01*ms, .1*ms, 5*ms, 20*ms])
    
    criterion = LpError(p=p, varname='V')
    
    if best_params is None:
        results = modelfitting( model = equations,
                                reset = reset,
                                threshold = threshold,
                                data = trace,
                                input = current,
                                cpu = cpu,
                                gpu = gpu,
                                dt = dt,
                                popsize = popsize,
                                maxiter = maxiter,
                                onset = overlap,
                                criterion = criterion,
                                initial_values = initial_values,
                                **params
                                )
        print_table(results)
        best_params = results.best_params
    else:
        results = best_params
        for key in best_params.keys():
            best_params[key] = [best_params[key]]
    
    criterion_values, record_values = simulate( model = equations,
                                                reset = reset,
                                                threshold = threshold,
                                                data = trace,
                                                input = current,
                                                dt = dt,
                                                neurons = slices, # 1 neuron/slice
                                                onset = overlap,
                                                criterion = criterion,
                                                use_gpu = True,
                                                record = record,
                                                initial_values = initial_values,
                                                **best_params
                                                )
    
    traceV = record_values[0][:,int(overlap/dt):].flatten()
    traceVe = record_values[1][:,int(overlap/dt):].flatten()
    
    
#    return traceV, traceVe, results
    return trace0 - traceVe, traceV, traceVe, results
