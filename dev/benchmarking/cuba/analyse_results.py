from numpy import *
from pylab import *
import pickle
import os
import itertools
import cuba_runopts

basedir = 'data'

def get_packaged_results(packagename, var, desc):
    if basedir: packagename = basedir+'/'+packagename
    exec open(packagename+'.py').read()
    res = locals()[var]
    return (res,desc)

def get_brian(ext, desc, runtype=''):
    if ext:
        ext = '_'+ext
    if runtype:
        runtype = '_'+runtype
    pre = ''
    if basedir: pre = basedir+'/'
    inputfile = open(pre+'brian_cuba'+runtype+'_results'+ext+'.pkl','rb')
    brian_cuba = pickle.load(inputfile)
    inputfile.close()
    return (brian_cuba,desc)

def plotresults(res, desc):
    N, t, spikes, results = zip(*res)
    R = array(results)
    tstd = std(R,axis=1)[:,0]    
    errorbar(N,t,yerr=tstd, label=desc)

def plotcomparativeresults(base, res, desc):
    Nb, tb, spikesb, resultsb = zip(*base)
    N, t, spikes, results = zip(*res)
    tb = array(tb)[:len(N)]
    plot(N,array(t)/tb, '-o',label=desc)

def plotrates(res, desc):
    N, t, spikes, results = zip(*res)
    R = array(results)
    tstd = std(R,axis=1)[:,1]    
    errorbar(N,spikes,yerr=tstd, label=desc)

def plotvaryconresults(res, desc):
    N, t, spikes, results = zip(*res)
    results = array(results)
    results[:,:,1]/=N[0]*cuba_runopts.duration
    spikes = array(spikes)/(N[0]*cuba_runopts.duration)
    for k, n in zip(results, N):
        R = array(k)
        plot(R[:,1], R[:,0], '.', label=desc)
    R = array(results)
    ststd = std(R,axis=1)
    sstd = ststd[:,1]
    tstd = ststd[:,0]
    errorbar(spikes, t, tstd, sstd, label=desc)

def plotvaryweresults(res, desc):
    N, t, spikes, results = zip(*res)
    results = list(itertools.chain(*results))
    results = array(results)
    results[:,1]/=N[0]*cuba_runopts.duration
    plot(results[:,1], results[:,0], '.', label=desc)        

def do_analysis(all_results, showrates=True):
    cpp_cuba = all_results[0][0]
    cpp_cuba_desc = all_results[0][1]
    N, t, spikes, results = zip(*cpp_cuba)
    Nmax = max(N)
    Nmin = min(N)
    
    d1 = 1
    if showrates:
        d2 = 3
    else:
        d2 = 2
    
    figure()
    
    subplot(d1,d2,1)
    for r, desc in all_results:
        plotresults(r, desc)
    #legend([desc for r,desc in all_results])
    #legend(loc='upper left')
    axis(ymin=0, xmin=Nmin, xmax=Nmax)
    xlabel('N')
    ylabel('Time (s)')
    
    subplot(d1,d2,2)
    for r, desc in all_results:
        plotcomparativeresults(cpp_cuba, r, desc)
    #legend([desc for r,desc in all_results])
    #legend(loc='upper left')
    legend()
    axis(ymin=0, xmin=Nmin, xmax=Nmax)
    xlabel('N')
    ylabel('Time relative to '+cpp_cuba_desc+' (s)')
    
    if showrates:
        subplot(d1,d2,3)
        for r, desc in all_results:
            plotrates(r, desc)
        axis(ymin=0, xmin=Nmin, xmax=Nmax)
        xlabel('N')
        ylabel('Num. spikes')
    
#    if showrates:
#        subplot(d1,d2,4)
#    else:
#        subplot(d1,d2,3)
#    figlegend()
    
#    show()

def do_analysis_varywe(all_results):
    figure()
    for res, desc in all_results:
        plotvaryweresults(res, desc)
    xlabel('Firing rate per neuron (Hz)')
    ylabel('Time (s)')
    legend(loc='upper left')
#    do_analysis(all_results)

def cd(s):
    global basedir
    basedir = s

if __name__=='__main__':
    all_results = []
    all_results.append(get_packaged_results('cpp_cuba_results_matrix','cpp_cuba','C++ (gen. matrix)'))
    all_results.append(get_packaged_results('cpp_cuba_results','cpp_cuba','C++ (Euler)'))
    all_results.append(get_packaged_results('cpp_cuba_results_matrix_efficient','cpp_cuba','C++ (fix. matrix)'))
    all_results.append(get_packaged_results('matlab_cuba_results','matlab_cuba','Matlab'))
    all_results.append(get_brian('compile_nounits','Brian compiled, no units'))
    do_analysis(all_results, showrates=False)
    
    all_results = []
    all_results.append(get_packaged_results('cpp_cuba_nospiking_results_matrix','cpp_cuba','C++ (gen. matrix)'))
    all_results.append(get_packaged_results('cpp_cuba_nospiking_results','cpp_cuba','C++ (Euler)'))
    all_results.append(get_packaged_results('matlab_cuba_nospiking_results','matlab_cuba','Matlab'))
    all_results.append(get_brian('compile_nounits','Brian compiled, no units','nospiking'))
    do_analysis(all_results, showrates=False)

#    all_results = []
#    all_results.append(get_packaged_results('cpp_cuba_varywe_results_matrix','cpp_cuba','C++ (gen. matrix)'))
#    all_results.append(get_packaged_results('cpp_cuba_varywe_results','cpp_cuba','C++ (Euler)'))
#    all_results.append(get_packaged_results('matlab_cuba_varywe_results','matlab_cuba','Matlab'))
#    all_results.append(get_brian('compile_nounits_1000','Brian','varywe'))
#    do_analysis_varywe(all_results)

    all_results = []
    all_results.append(get_packaged_results('cpp_cuba_varywe_results_matrix_4000','cpp_cuba','C++ (gen. matrix)'))
    all_results.append(get_packaged_results('matlab_cuba_varywe_results_4000','matlab_cuba','Matlab'))
    all_results.append(get_brian('compile_nounits_4000','Brian','varywe'))
    do_analysis_varywe(all_results)

    show()