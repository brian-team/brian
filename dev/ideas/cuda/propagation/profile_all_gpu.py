from brian import *
from brian.experimental.codegen2 import *
import numpy
import time
import numpy.random as nrandom
import random as prandom
from mpl_toolkits.mplot3d import Axes3D

#log_level_info()

##### PROFILING CODE

def profile(gpu_module, Nsource, Ntarget, Nstate, complexity, sparseness,
            rate, parameters={}, target_time=1.0,
            scalar='double',
            dothreshreset=True,
            doconn=True,
            ):
    clear(True, True)
    reinit_default_clock()
    use_gpu = bool(gpu_module)
    if use_gpu:
        ns = {}
        exec 'from %s import GPUConnection'%gpu_module in ns
        GPUConnection = ns['GPUConnection']
        Conn = GPUConnection
        language = GPULanguage(force_sync=False,
                               scalar=scalar,
                               )
    else:
        Conn = Connection
        language = CLanguage()
        parameters = {}

    nrandom.seed(3212212)
    prandom.seed(3432334)

    expr = 'X'
    for i in xrange(complexity):
        expr = 'sin({expr}+X)'.format(expr=expr)
    eqs = '''
    dv/dt = {rate}/second : 1
    '''.format(rate=float(rate))
    for i in xrange(Nstate):
        eqs += 'dX/dt = {expr}/(10*ms) : 1\n'.format(expr=expr).replace('X', 'x'+str(i))
    
    #source = PoissonGroup(Nsource, rate)
    reset = 'v = 0'
    threshold = 'v>1'
    if dothreshreset:
        source = NeuronGroup(Nsource, eqs,
                             reset=reset,
                             threshold=threshold
                             )
    else:
        source = NeuronGroup(Nsource, eqs)
    source.v = rand(Nsource)
    source._state_updater = CodeGenStateUpdater(source, euler, language, clock=source.clock)
    if dothreshreset:
        source._threshold = CodeGenThreshold(source, threshold, language)
        source._resetfun = CodeGenReset(source, reset, language)
    target = NeuronGroup(Ntarget, 'dV/dt=0/second:1')
    target._state_updater = CodeGenStateUpdater(target, euler, language, clock=target.clock)

    if doconn:
        C = Conn(source, target, 'V', weight=1, sparseness=sparseness, **parameters)

    run(10*ms)

    @network_operation
    def check_done():
        if time.time()-start>target_time:
            stop()
    
    start = time.time()
    run(1e10*second)
#    if use_gpu:
#        language.gpu_man.copy_to_host(True)
    end = time.time()
    elapsed = defaultclock.t
    realtime_ratio = (end-start)/float(elapsed)

    summary = ''
    if use_gpu:
        summary += 'Using GPU (algorithm %s)\n'%gpu_module
        for key, value in parameters.items():
            summary += '    '+key+': '+str(value)+'\n'
    else:
        summary += 'Using CPU\n'
    summary += 'Nsource={Nsource}, Ntarget={Ntarget}, sparseness={sparseness}, rate={rate}\n'.format(
        Nsource=Nsource, Ntarget=Ntarget, sparseness=sparseness, rate=rate)
    summary += 'Ratio compared to realtime: %.3f'%realtime_ratio
    
    return realtime_ratio, summary

if __name__=='__main__':
    if 1:
        log_level_info()
        ratio, summary = profile(
            '',
#            'vectorise_over_postsynaptic_offset',
#            'double_vectorise_over_spsyn_targetidx_blocked',
            #scalar='float',
            dothreshreset=True,
            doconn=True,
            Nsource=10000, Ntarget=10000, sparseness=0.1, rate=10*Hz,
            Nstate=3, complexity=2,
            parameters=dict(),
            )
        print summary
    if 0:
        show_3d_plots = True
        show_individual_2d_plots = True # only used if 3D plots not used
        repeats = 1
        bases = {
            'C++':('b', dict(gpu_module='')),
            'CUDA/VPO':('g', dict(gpu_module='vectorise_over_postsynaptic_offset',
                                  parameters=dict())),
            'CUDA/VSS':('r', dict(gpu_module='vectorise_over_spiking_synapses',
                                  parameters=dict())),
#            'CUDA/VSS/NA':((0.5,0,0), dict(gpu_module='vectorise_over_spiking_synapses',
#                                           parameters=dict(use_atomic=False))),
            'CUDA/DVPOT':('y', dict(gpu_module='double_vectorise_over_postsynoff_targetidx_blocked',
                                    parameters=dict())),
#            'CUDA/DVPOT/MW':((0.5,0.5,0), dict(gpu_module='double_vectorise_over_postsynoff_targetidx_blocked',
#                                               parameters=dict(masked_write=False))),
            'CUDA/DVSST':('m', dict(gpu_module='double_vectorise_over_spsyn_targetidx_blocked',
                                    parameters=dict())),
            'CUDA/DVSST/MW':((0.5,0,0.5), dict(gpu_module='double_vectorise_over_spsyn_targetidx_blocked',
                                               parameters=dict(masked_write=False))),
            }
        plot_params = [
            #dict(Nsource=100, Ntarget=100000, sparseness=0.1,
            #     spikesperdt=[1, 5, 10, 50, 100],
            #     ),
            #dict(Nsource=100, Ntarget=100000, spikesperdt=5,
            #     sparseness=[0.001, 0.01, 0.1, 0.2],
            #     ),
            dict(Nsource=100, Ntarget=100000,
                 # low detail
#                 spikesperdt=[1, 10, 50, 100],
#                 sparseness=[0.001, 0.01, 0.1, 0.3],
                 # medium detail
#                 spikesperdt=[1, 3, 5, 10, 25, 50],
#                 sparseness=[0.001, 0.005, 0.01, 0.05, 0.1],
                 # high detail
                 spikesperdt=range(1,10)+range(10, 101, 10),
                 sparseness=exp(linspace(log(0.001), log(0.3), 20)),
                 ),
            ]
        start_time = time.time()
        for pp in plot_params:
            basekwds = pp
            varying = {}
            unvarying = []
            images = {}
            for k, v in pp.items():
                if isinstance(v, (list, tuple, ndarray)):
                    varying[k] = v
                else:
                    unvarying.append(k+'='+str(v))
            unvarying = ', '.join(unvarying)
            if len(varying)>2 or len(varying)==0:
                print 'Can only vary one or two parameters at a time.'
                exit()
            if len(varying)==1:
                (varykey, varyval), = varying.items()
                figure()
                title(unvarying)
            elif len(varying)==2:
                (varykey1, varyval1), (varykey2, varyval2) = varying.items()
                if show_3d_plots:
                    f = figure()
                    title(unvarying)
                    ax = f.add_subplot(111, projection='3d')
            for ibase, (name, (col, kwds)) in enumerate(bases.items()):
                kwds.update(basekwds)
                if len(varying)==1:
                    T = []
                    for v in varyval:
                        kwds[varykey] = v
                        allt = []
                        for _ in xrange(repeats):
                            t, summary = profile_gpu_connection(**kwds)
                            print summary
                            allt.append(t)
                        t = mean(allt)
                        T.append(t)
                    plot(varyval, T, '-o', label=name, color=col)
                elif len(varying)==2:
                    allT = []
                    for v1 in varyval1:
                        T = []
                        for v2 in varyval2:
                            kwds[varykey1] = v1
                            kwds[varykey2] = v2
                            allt = []
                            for _ in xrange(repeats):
                                t, summary = profile_gpu_connection(**kwds)
                                print summary
                                allt.append(t)
                            t = mean(allt)
#                            t = (-1)**ibase*((v1/amax(varyval1))**2+(v2/amax(varyval2))**2)+ibase
                            T.append(t)
                        allT.append(T)
                    I = array(allT).T
                    images[name] = I
                    if show_3d_plots:
                        V1, V2 = meshgrid(arange(len(varyval1)), arange(len(varyval2)))
                        ax.plot_surface(V1, V2, I, color=col, label=name,
                                        alpha=0.7, rstride=1, cstride=1, shade=False)
                    elif show_individual_2d_plots:
                        figure()
                        title(unvarying+'\n'+name)
                        #imshow(I, origin='lower left', aspect='auto', interpolation='nearest')
                        pcolor(I) # works better over ssh
                        colorbar()
                        idx = array(xticks()[0], dtype=int)
                        idx = unique(idx[idx<len(varyval1)])
                        idx = idx[idx>=0]
                        xticks(idx+0.5, array(varyval1)[idx])
                        idx = array(yticks()[0], dtype=int)
                        idx = unique(idx[idx<len(varyval2)])
                        idx = idx[idx>=0]
                        yticks(idx+0.5, array(varyval2)[idx])
                        xlabel(varykey1)
                        ylabel(varykey2)
            if len(varying)==1:
                xlabel(varykey)
                ylabel('Ratio compared to realtime')
                legend()
            elif len(varying)==2:
                if show_3d_plots:
                    xlabel(varykey1)
                    ylabel(varykey2)
                    idx = array(ax.get_xticks(), dtype=int)
                    idx = unique(idx[idx<len(varyval1)])
                    if len(idx)==0:
                        idx = arange(len(varyval1))
                    ax.set_xticks(idx)
                    ax.set_xticklabels(map(str, array(varyval1)[idx]))
                    idx = array(ax.get_yticks(), dtype=int)
                    idx = unique(idx[idx<len(varyval2)])
                    if len(idx)==0:
                        idx = arange(len(varyval2))
                    ax.set_yticks(idx)
                    ax.set_yticklabels(map(str, array(varyval2)[idx]))
                    #legend()
                figure()
                title('Best algorithm')
                subplot(111)
                for i in xrange(len(varyval1)):
                    for j in xrange(len(varyval2)):
                        best = 1e10
                        c = 'k'
                        for name, I in images.items():
                            if I[j, i]<best:
                                best = I[j, i]
                                c = bases[name][0]
                                fill([i,i+1,i+1,i], [j,j,j+1,j+1], color=c)
                idx = array(xticks()[0], dtype=int)
                idx = unique(idx[idx<len(varyval1)])
                idx = idx[idx>=0]
                xticks(idx+0.5, array(varyval1)[idx])
                idx = array(yticks()[0], dtype=int)
                idx = unique(idx[idx<len(varyval2)])
                idx = idx[idx>=0]
                yticks(idx+0.5, array(varyval2)[idx])
                xlabel(varykey1)
                ylabel(varykey2)
        print 'Finished, took %ds'%int(time.time()-start_time)
        show()
        
