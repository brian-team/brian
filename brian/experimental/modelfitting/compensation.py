from brian import *
from modelfitting import *
import time, os

dt = defaultclock.dt

def compensate(current, trace, popsize = 1000, maxiter = 2,
               equations = None, reset = None, threshold = None,
                  slice_duration = None, overlap = 0*ms,
                  initial = None, dt = defaultclock.dt,
                  cpu = None, gpu = 1, record=['V','Ve'],
                  cut = None, cut_length = 0.0,
                  p = 1., results=None,
                  **params):
    
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
            V=V0+Ve : 1
            dV0/dt=(R*I-V0+Vr)/tau : 1
            dVe/dt=(Re*I-Ve)/taue : 1
            I : 1
            R : 1
            tau : second
            taue : second
            Re : 1
            Vr : 1
        ''')
    if threshold is None:
        threshold = """
                    V>100000
                    """
    if reset is None:
        reset = ""
    
    criterion = LpError(p=p, varname='V')
    
    if results is None:
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
    
    try:
        best_params = results.best_params
    except:
        best_params = results.best_pos
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
    
    return traceV, traceVe, results

if __name__ == '__main__':
    
    current = loadtxt('current.txt')
    trace = loadtxt('trace_artificial.txt')
    
    params = dict(  R = [1e6, 1e9, 1e10, 1e10],
                    Re = [0,0,0.1,0.1],
                    Vr = [0., 1.],
                    tau =  [1*ms, 5*ms, 50*ms, 200*ms],
                    taue = [1*ms, 1*ms, 2*ms, 2*ms])
    
    traceV, traceVe, results = compensate(current, trace, threshold=1,reset=0,
                                            popsize=100, maxiter=1,
                                            **params)
    print traceV.shape
    print traceVe.shape
    
    plot(trace)
    plot(traceV)
    plot(trace-traceVe)
    grid()
    show()
    
    