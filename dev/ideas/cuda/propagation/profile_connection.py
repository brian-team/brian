from brian import *
from brian.experimental.codegen2 import *
import numpy
import time
import numpy.random as nrandom
import random as prandom
from mpl_toolkits.mplot3d import Axes3D
import joblib
from matplotlib import cm, rcParams

#log_level_info()

##### PROFILING CODE

cache_mem = joblib.Memory(cachedir='/home/dan/programming/joblib_cache', verbose=0)

@cache_mem.cache
def profile_gpu_connection(gpu_module, Nsource, Ntarget, sparseness, spikesperdt,
                           parameters={}, target_time=1.0,
                           arbitrary=None):
    clear(True, True)
    reinit_default_clock()
    use_gpu = bool(gpu_module)
    if use_gpu:
        ns = {}
        exec 'from %s import GPUConnection'%gpu_module in ns
        GPUConnection = ns['GPUConnection']
        Conn = GPUConnection
        language = GPULanguage(force_sync=False)
    else:
        Conn = Connection
        language = CLanguage()
        parameters = {}

    nrandom.seed(3212212)
    prandom.seed(3432334)

    rate = spikesperdt/(Nsource*defaultclock.dt)
        
    source = PoissonGroup(Nsource, rate)
    target = NeuronGroup(Ntarget, 'dv/dt=0/second:1')
    target._state_updater = CodeGenStateUpdater(target, euler, language, clock=target.clock)

    C = Conn(source, target, 'v', weight=1, sparseness=sparseness, **parameters)

    run(10*ms)

    @network_operation
    def check_done():
        if time.time()-start>target_time:
            stop()
    
    start = time.time()
    run(1e10*second)
    if use_gpu:
        language.gpu_man.copy_to_host(True)
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
    summary += 'Nsource={Nsource}, Ntarget={Ntarget}, sparseness={sparseness}, spikesperdt={spikesperdt}\n'.format(
        Nsource=Nsource, Ntarget=Ntarget, sparseness=sparseness, spikesperdt=spikesperdt)
    summary += 'Ratio compared to realtime: %.3f'%realtime_ratio
    
    return realtime_ratio, summary

if __name__=='__main__':
    if 0:
        ratio, summary = profile_gpu_connection(
            'vectorise_over_postsynaptic_offset', True,
            Nsource=4000, Ntarget=4000, sparseness=0.02, spikesperdt=2,
            parameters=dict(
                use_atomic=False,
                ),
            )
        print summary
    if 1:
        show_3d_plots = False
        show_individual_2d_plots = False # only used if 3D plots not used
        repeats = 1
#        bases = {
##            'C++':('b', dict(gpu_module='')),
#            'CUDA/VPO':('g', dict(gpu_module='vectorise_over_postsynaptic_offset',
#                                  parameters=dict())),
##            'CUDA/VPO/F':((0.5, 1.0, 0.5),
##                          dict(gpu_module='vectorise_over_postsynaptic_offset',
##                               parameters=dict(use_float=True))),
#            'CUDA/VSS':('r', dict(gpu_module='vectorise_over_spiking_synapses',
#                                  parameters=dict())),
##            'CUDA/VSS/F':((1.0, 0.5, 0.5),
##                          dict(gpu_module='vectorise_over_spiking_synapses',
##                               parameters=dict(use_float=True))),
##            'CUDA/VSS/NA':((0.5,0,0), dict(gpu_module='vectorise_over_spiking_synapses',
##                                           parameters=dict(use_atomic=False))),
#            'CUDA/DVPOT':('y', dict(gpu_module='double_vectorise_over_postsynoff_targetidx_blocked',
#                                    parameters=dict())),
##            'CUDA/DVPOT/F':((1,1,.5),
##                            dict(gpu_module='double_vectorise_over_postsynoff_targetidx_blocked',
##                                 parameters=dict(use_float=True))),
#            'CUDA/DVPOT/MW':((0.5,0.5,0), dict(gpu_module='double_vectorise_over_postsynoff_targetidx_blocked',
#                                               parameters=dict(masked_write=False))),
##            'CUDA/DVPOT/MW':((0.75,0.75,0.5),
##                             dict(gpu_module='double_vectorise_over_postsynoff_targetidx_blocked',
##                                  parameters=dict(masked_write=False,
##                                                  use_float=True))),
#            'CUDA/DVSST':('m', dict(gpu_module='double_vectorise_over_spsyn_targetidx_blocked',
#                                    parameters=dict())),
##            'CUDA/DVSST/F':((1,0.5,1),
##                            dict(gpu_module='double_vectorise_over_spsyn_targetidx_blocked',
##                                 parameters=dict(use_float=True))),
#            'CUDA/DVSST/MW':((0.5,0,0.5),
#                             dict(gpu_module='double_vectorise_over_spsyn_targetidx_blocked',
#                                  parameters=dict(masked_write=False))),
##            'CUDA/DVSST/MW/F':((0.75,0.5,0.75),
##                               dict(gpu_module='double_vectorise_over_spsyn_targetidx_blocked',
##                                    parameters=dict(masked_write=False,
##                                                    use_float=True))),
#            }

        bases = {
            'CUDA/VPO':('g', dict(gpu_module='vectorise_over_postsynaptic_offset',
                                  parameters=dict())),
            'CUDA/VSS':('r', dict(gpu_module='vectorise_over_spiking_synapses',
                                  parameters=dict())),
            'CUDA/DVPOT':('y', dict(gpu_module='double_vectorise_over_postsynoff_targetidx_blocked',
                                    parameters=dict())),
#            'CUDA/DVPOT/MW':((0.5,0.5,0), dict(gpu_module='double_vectorise_over_postsynoff_targetidx_blocked',
#                                               parameters=dict(masked_write=False))),
            'CUDA/DVSST':('m', dict(gpu_module='double_vectorise_over_spsyn_targetidx_blocked',
                                    parameters=dict())),
#            'CUDA/DVSST/MW':((0.5,0,0.5),
#                             dict(gpu_module='double_vectorise_over_spsyn_targetidx_blocked',
#                                  parameters=dict(masked_write=False))),
            }
        repeats = 10
        target_time = 1.0
#        repeats = 1
#        target_time = 0.1
        plot_params = [
            dict(Nsource=100, Ntarget=100000,
                 # low detail
#                 spikesperdt=[1, 10, 50, 100],
#                 sparseness=[0.001, 0.01, 0.1, 0.3],
                 # medium detail
#                 spikesperdt=[1, 3, 5, 10, 25, 50],
#                 sparseness=[0.001, 0.005, 0.01, 0.05, 0.1],
                 # high detail
#                 spikesperdt=range(1,10)+range(10, 101, 10),
#                 sparseness=exp(linspace(log(0.001), log(0.3), 20)),
                # low detail, large range
#                spikesperdt=[1, 10, 50, 100],
#                sparseness=[0.05, 0.1, 0.3, 0.4, 0.5],
                # medium detail, large range
#                 spikesperdt=range(2, 10, 2)+range(10, 101, 20),
#                 sparseness=exp(linspace(log(0.001), log(0.8), 11)),
#                 sparseness=hstack((exp(linspace(log(0.001), log(0.1), 6)),
#                                    linspace(0.1, 0.8, 5)
#                                    )),
                # medium detail linear sparseness, large range
#                 spikesperdt=range(1, 10, 2)+range(10, 101, 20),
#                 sparseness=linspace(0.001, 0.5, 10),
                # high detail, large range
                 spikesperdt=range(1, 10)+range(10, 101, 10),
                 sparseness=exp(linspace(log(0.001), log(0.8), 19)),
                 ),
            ]
        labels = {'spikesperdt': 'Spikes per timestep',
                  'sparseness': 'Sparseness (log scale)',
                  }
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
                        for rep in xrange(repeats):
                            kwds['arbitrary'] = rep
                            kwds['target_time'] = target_time
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
                            for rep in xrange(repeats):
                                kwds['arbitrary'] = rep
                                kwds['target_time'] = target_time
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
                        xlabel(labels.get(varykey1, varykey1))
                        ylabel(labels.get(varykey2, varykey2))
            if len(varying)==1:
                xlabel(labels.get(varykey, varykey))
                ylabel('Ratio compared to realtime')
                legend()
            elif len(varying)==2:
                if show_3d_plots:
                    xlabel(labels.get(varykey1, varykey1))
                    ylabel(labels.get(varykey2, varykey2))
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
                rcParams['font.size'] = 10
                rcParams['figure.subplot.left']  = 0.1
                rcParams['figure.subplot.right'] = .98
                rcParams['figure.subplot.bottom'] = .12
                rcParams['figure.subplot.top'] = .85
                rcParams['figure.subplot.wspace'] = 0.0
                rcParams['figure.subplot.hspace'] = 0.5
                figure(figsize=(10, 3))
                subplot(131)
                text(-0.15, 1.1, 'A', size=14, transform=gca().transAxes)
                title('Best algorithm')
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
                idx = hstack((idx, len(varyval1)-1))
                idx = unique(idx[idx<len(varyval1)])
                idx = idx[idx>=0]
                idx_x = idx
                xticks(idx+0.5, array(varyval1)[idx])
                idx = array(yticks()[0], dtype=int)
                idx = hstack((idx, len(varyval2)-1))
                idx = unique(idx[idx<len(varyval2)])
                idx = idx[idx>=0]
                #yticks(idx+0.5, array(varyval2)[idx])
                yticks(idx+0.5, ['%.3f'%y for y in array(varyval2)[idx]])
                xlabel(labels.get(varykey1, varykey1))
                ylabel(labels.get(varykey2, varykey2))
                axis('tight')
                gca().set_aspect('equal')
                
                ratios_w2b = zeros((len(varyval2), len(varyval1)))
                ratios_s2f = zeros((len(varyval2), len(varyval1)))
                for i in xrange(len(varyval1)):
                    for j in xrange(len(varyval2)):
                        best = 1e10
                        worst = 0
                        c = 'k'
                        allI = []
                        for name, I in images.items():
                            allI.append(I[j, i])
                        best = amin(allI)
                        worst = amax(allI)
                        ratios_w2b[j, i] = worst/best
                        allI.sort()
                        ratios_s2f[j, i] =  allI[1]/allI[0]
                        
                subplot(132)
                text(-0.15, 1.1, 'B', size=14, transform=gca().transAxes)
                title('Ratio worst:best')
#                    title('Ratio second best:best')
                imshow(ratios_w2b, origin='lower left', aspect='auto',
                       interpolation='nearest',
#                       vmin=1, vmax=amax(ratios_w2b),
                       extent=(0, len(varyval1),
                               0, len(varyval2)))
                axis('tight')
                colorbar()
                xticks(idx_x+0.5, array(varyval1)[idx_x])
                yticks(idx+0.5, ['' for _ in idx])#array(varyval2)[idx])
                xlabel(labels.get(varykey1, varykey1))
                #ylabel(labels.get(varykey2, varykey2))
                gca().set_aspect('equal')

                subplot(133)
                text(-0.15, 1.1, 'C', size=14, transform=gca().transAxes)
                title('Ratio second best:best')
                imshow(ratios_s2f, origin='lower left', aspect='auto',
                       interpolation='nearest',
                       #vmin=1, vmax=amax(ratios_w2b),
                       extent=(0, len(varyval1),
                               0, len(varyval2)))
                axis('tight')
                colorbar()
                xticks(idx_x+0.5, array(varyval1)[idx_x])
                yticks(idx+0.5, ['' for _ in idx])#array(varyval2)[idx])
                xlabel(labels.get(varykey1, varykey1))
                #ylabel(labels.get(varykey2, varykey2))
                gca().set_aspect('equal')
        print 'Finished, took %ds'%int(time.time()-start_time)
        show()
        
