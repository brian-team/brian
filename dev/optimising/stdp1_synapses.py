'''
stdp1_bis / synapses: 260s (this version)
stdp1 / synapses: 245s
stdp1 / connection / c: 121s
stdp1 / connection / python stdp, c propagate: 175s
stdp1 / connection / python stdp, propagate: 178s

Partial profiling results:

        1    0.000    0.000  306.769  306.769 C:\Users\Dan\programming\brian\dev\optimising\stdp1_synapses.py:47(f)
  1000000    6.584    0.000  152.892    0.000 C:\Users\Dan\programming\brian\brian\synapses\synapses.py:707(update)
  2000000    8.203    0.000   88.519    0.000 C:\Users\Dan\programming\brian\brian\neurongroup.py:474(update)
  1000000    1.859    0.000   63.485    0.000 C:\Users\Dan\programming\brian\brian\directcontrol.py:438(update)
  1000000   14.161    0.000   55.861    0.000 C:\Users\Dan\programming\brian\brian\threshold.py:463(__call__)
  2000000    3.194    0.000   45.535    0.000 C:\Users\Dan\programming\brian\brian\connections\connection.py:295(do_propagate)
  2000000    5.157    0.000   34.137    0.000 C:\Users\Dan\programming\brian\brian\synapses\spikequeue.py:323(propagate)
  1000000   19.955    0.000   19.955    0.000 {method 'rand' of 'mtrand.RandomState' objects}
   780552   18.013    0.000   18.195    0.000 C:\Users\Dan\programming\brian\brian\synapses\spikequeue.py:287(insert_homogeneous)
  1501622    3.357    0.000   14.722    0.000 C:\Python27\lib\site-packages\numpy\lib\function_base.py:1258(extract)
  1000000    5.420    0.000   14.233    0.000 C:\Users\Dan\programming\brian\brian\threshold.py:153(__call__)
   780552    0.817    0.000    9.802    0.000 C:\Python27\lib\site-packages\numpy\core\shape_base.py:228(hstack)
  3003244    2.292    0.000    7.782    0.000 C:\Python27\lib\site-packages\numpy\core\fromnumeric.py:1044(ravel)
  7561480    5.508    0.000    5.508    0.000 {method 'take' of 'numpy.ndarray' objects}

Analysis:

Numbers are percentage of time spent in that function/class, indented numbers
give times as a percentage of total (left) and of parent (right)

50%    Synapses.update
    5%       9%    numpy.extract
    2%       4%    numpy.take
15%    Connection.do_propagate
   11%      73%    SpikeQueue.propagate
    6%                 53%      SpikeQueue.insert_homogeneous
29%    NeuronGroup.update
   21%      72%    PoissonGroup
   18%                  87%    PoissonThreshold
    6%                            35%    rand()
            16%    Threshold.__call__

Synapses together comprises 65% of the time while NeuronGroup (which is the same
in all versions) comprises 15%, and another 5% is also presumably more or less
the same in both. From this we can deduce that Synapses is the following amount
slower than:

Version                                              Faster than Synapses

stdp1 / connection / c:                              3x
stdp1 / connection / python stdp, c propagate:       1.69x
stdp1 / connection / python stdp, propagate: 178s    1.65x

Based on stdp1_bis taking 260s of which 20% is shared, i.e. 52s. So compute
amount faster by (time_with_synapses-52)/(time_with_alternative-52)

Note that the argsort, extract, take routines take less than 8% of the total
time, with the argsort itself not even appearing (i.e. taking less than 0.5%
of the total time).

Complete profiling results:

         134570775 function calls (134569741 primitive calls) in 204.547 seconds

   Ordered by: cumulative time, call count
   List reduced from 265 to 50 due to restriction <50>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000  306.769  306.769 <string>:1(<module>)
        1    0.000    0.000  306.769  306.769 C:\Users\Dan\programming\brian\dev\optimising\stdp1_synapses.py:47(f)
        1    0.000    0.000  306.769  306.769 C:\Users\Dan\programming\brian\brian\network.py:892(run)
        1    2.432    2.432  306.626  306.626 C:\Users\Dan\programming\brian\brian\network.py:532(run)
  1000000    5.179    0.000  302.700    0.000 C:\Users\Dan\programming\brian\brian\network.py:507(update)
  1000000    6.584    0.000  152.892    0.000 C:\Users\Dan\programming\brian\brian\synapses\synapses.py:707(update)
  2000000    8.203    0.000   88.519    0.000 C:\Users\Dan\programming\brian\brian\neurongroup.py:474(update)
  1000000    1.859    0.000   63.485    0.000 C:\Users\Dan\programming\brian\brian\directcontrol.py:438(update)
  1000000   14.161    0.000   55.861    0.000 C:\Users\Dan\programming\brian\brian\threshold.py:463(__call__)
  2000000    3.194    0.000   45.535    0.000 C:\Users\Dan\programming\brian\brian\connections\connection.py:295(do_propagate)
  2000000    5.157    0.000   34.137    0.000 C:\Users\Dan\programming\brian\brian\synapses\spikequeue.py:323(propagate)
  1000000   19.955    0.000   19.955    0.000 {method 'rand' of 'mtrand.RandomState' objects}
   780552   18.013    0.000   18.195    0.000 C:\Users\Dan\programming\brian\brian\synapses\spikequeue.py:287(insert_homogeneous)
  1501622    3.357    0.000   14.722    0.000 C:\Python27\lib\site-packages\numpy\lib\function_base.py:1258(extract)
  1000000    5.420    0.000   14.233    0.000 C:\Users\Dan\programming\brian\brian\threshold.py:153(__call__)
  1000000    1.409    0.000   10.055    0.000 C:\Users\Dan\programming\brian\brian\clock.py:194(<lambda>)
   780552    0.817    0.000    9.802    0.000 C:\Python27\lib\site-packages\numpy\core\shape_base.py:228(hstack)
  3279118    9.105    0.000    9.105    0.000 {method 'nonzero' of 'numpy.ndarray' objects}
  1000271    3.453    0.000    8.649    0.000 C:\Users\Dan\programming\brian\brian\fundamentalunits.py:559(with_dimensions)
  3000000    2.613    0.000    8.607    0.000 C:\Users\Dan\programming\brian\brian\neurongroup.py:524(reset)
  3009356    6.773    0.000    8.319    0.000 C:\Users\Dan\programming\brian\brian\unitsafefunctions.py:81(f)
  2000000    2.892    0.000    8.203    0.000 C:\Users\Dan\programming\brian\brian\neurongroup.py:500(get_spikes)
  3003244    2.292    0.000    7.782    0.000 C:\Python27\lib\site-packages\numpy\core\fromnumeric.py:1044(ravel)
  1000000    5.295    0.000    7.380    0.000 C:\Users\Dan\programming\brian\brian\stateupdater.py:398(__call__)
   780594    1.233    0.000    7.359    0.000 {map}
19632964/19631963    4.345    0.000    6.758    0.000 {len}
  1504678    2.925    0.000    6.126    0.000 C:\Python27\lib\site-packages\numpy\core\shape_base.py:6(atleast_1d)
  1000000    3.726    0.000    5.701    0.000 C:\Users\Dan\programming\brian\brian\reset.py:139(__call__)
  7561480    5.508    0.000    5.508    0.000 {method 'take' of 'numpy.ndarray' objects}
  2000010    2.611    0.000    5.053    0.000 C:\Users\Dan\programming\brian\brian\neurongroup.py:553(state_)
  1000000    2.881    0.000    4.970    0.000 C:\Python27\lib\site-packages\scipy\weave\inline_tools.py:133(inline)
  2000000    4.554    0.000    4.554    0.000 C:\Users\Dan\programming\brian\brian\synapses\spikequeue.py:207(peek)
  1000271    2.520    0.000    4.461    0.000 C:\Users\Dan\programming\brian\brian\fundamentalunits.py:554(__init__)
  3003304    1.827    0.000    3.967    0.000 C:\Python27\lib\site-packages\numpy\core\numeric.py:167(asarray)
  1504678    1.182    0.000    3.794    0.000 C:\Python27\lib\site-packages\numpy\core\fromnumeric.py:1322(clip)
  4507982    3.596    0.000    3.596    0.000 {numpy.core.multiarray.array}
  2000000    1.008    0.000    2.879    0.000 C:\Users\Dan\programming\brian\brian\utils\ccircular\ccircular.py:112(get_spikes)
  1504678    1.190    0.000    2.647    0.000 C:\Python27\lib\site-packages\numpy\core\numeric.py:237(asanyarray)
  1504678    2.612    0.000    2.612    0.000 {method 'clip' of 'numpy.ndarray' objects}
  2000000    2.345    0.000    2.503    0.000 C:\Users\Dan\programming\brian\brian\synapses\spikequeue.py:200(next)
  2000010    2.162    0.000    2.442    0.000 C:\Users\Dan\programming\brian\brian\group.py:46(state_)
  4000001    2.412    0.000    2.412    0.000 C:\Users\Dan\programming\brian\brian\group.py:37(__len__)
  2000000    1.048    0.000    2.401    0.000 C:\Users\Dan\programming\brian\brian\utils\ccircular\ccircular.py:109(push)
  8095487    2.243    0.000    2.243    0.000 {isinstance}
  1000000    2.085    0.000    2.085    0.000 {numpy.core._dotblas.dot}
  1501622    0.756    0.000    1.952    0.000 C:\Python27\lib\site-packages\numpy\core\fromnumeric.py:45(take)
  1000391    1.644    0.000    1.941    0.000 C:\Users\Dan\programming\brian\brian\fundamentalunits.py:172(__init__)
  2000000    1.871    0.000    1.871    0.000 {brian.utils.ccircular._ccircular.SpikeContainer_get_spikes}
  1000000    1.808    0.000    1.808    0.000 {apply}
  1501622    0.790    0.000    1.631    0.000 C:\Python27\lib\site-packages\numpy\core\fromnumeric.py:1129(nonzero)
Remaining functions take less than 0.5% of the total time each
  
Generated code for Synapses.update:

_post_neurons = _post.data.take(_synapses)
_perm = _post_neurons.argsort()
_aux = _post_neurons.take(_perm)
_flag = empty(len(_aux)+1, dtype=bool)
_flag[0] = _flag[-1] = 1
not_equal(_aux[1:], _aux[:-1], _flag[1:-1])
_F = _flag.nonzero()[0][:-1]
logical_not(_flag, _flag)
while len(_F):
    _u = _aux.take(_F)
    _i = _perm.take(_F)
    Apost[_synapses[_i]]=Apost[_synapses[_i]]*exp(-(-lastupdate[_synapses[_i]] + t)/taupost)
    Apre[_synapses[_i]]=Apre[_synapses[_i]]*exp(-(-lastupdate[_synapses[_i]] + t)/taupre)
    _target_ge[_u]+=w[_synapses[_i]]
    Apre[_synapses[_i]]+=dApre
    w[_synapses[_i]]=clip(w[_synapses[_i]]+Apost[_synapses[_i]],0,gmax)
    
    lastupdate[_synapses[_i]]=t
    
    _F += 1
    _F = extract(_flag.take(_F), _F)

Optimisation attempts:

Cache _synapses[_i]:

_post_neurons = _post.data.take(_synapses)
_perm = _post_neurons.argsort()
_aux = _post_neurons.take(_perm)
_flag = empty(len(_aux)+1, dtype=bool)
_flag[0] = _flag[-1] = 1
not_equal(_aux[1:], _aux[:-1], _flag[1:-1])
_F = _flag.nonzero()[0][:-1]
logical_not(_flag, _flag)
while len(_F):
    _u = _aux.take(_F)
    _i = _perm.take(_F)
    _synapses_i = _synapses[_i]
    Apost[_synapses_i]=Apost[_synapses_i]*exp(-(-lastupdate[_synapses_i] + t)/taupost)
    Apre[_synapses_i]=Apre[_synapses_i]*exp(-(-lastupdate[_synapses_i] + t)/taupre)
    _target_ge[_u]+=w[_synapses_i]
    Apre[_synapses_i]+=dApre
    w[_synapses_i]=clip(w[_synapses_i]+Apost[_synapses_i],0,gmax)
    
    lastupdate[_synapses_i]=t
    
    _F += 1
    _F = extract(_flag.take(_F), _F)
    
Time taken: 244s (2.8x, 1.6x, 1.5x slower than connection versions)

Do it directly and the old way (to gauge by the difference how much time the
fiddling around with argsort/etc. adds). We have to do this because the number
of spikes should be about the same for a fair comparison. First we compute the
time taken doing the direct operation, even though the computation is not
correct the total number of operations and memory accesses is the same, but in
a different order and with the messing around in Python removed.

_u = _post[_synapses]
_synapses_i = _synapses
#Apost[_synapses_i]=Apost[_synapses_i]*exp(-(-lastupdate[_synapses_i] + t)/taupost)
Apost[_synapses_i]*exp(-(-lastupdate[_synapses_i] + t)/taupost)
Apost[_synapses_i]
#Apre[_synapses_i]=Apre[_synapses_i]*exp(-(-lastupdate[_synapses_i] + t)/taupre)
Apre[_synapses_i]*exp(-(-lastupdate[_synapses_i] + t)/taupre)
Apre[_synapses_i]
#_target_ge[_u]+=w[_synapses_i]
_target_ge[_u]+w[_synapses_i]
_target_ge[_u]
#Apre[_synapses_i]+=dApre
Apre[_synapses_i]+dApre
Apre[_synapses_i]
#w[_synapses_i]=clip(w[_synapses_i]+Apost[_synapses_i],0,gmax)
clip(w[_synapses_i]+Apost[_synapses_i],0,gmax)
w[_synapses_i]
#lastupdate[_synapses_i]=t
lastupdate[_synapses_i]

_post_neurons = _post.data.take(_synapses)
_perm = _post_neurons.argsort()
_aux = _post_neurons.take(_perm)
_flag = empty(len(_aux)+1, dtype=bool)
_flag[0] = _flag[-1] = 1
not_equal(_aux[1:], _aux[:-1], _flag[1:-1])
_F = _flag.nonzero()[0][:-1]
logical_not(_flag, _flag)
while len(_F):
    _u = _aux.take(_F)
    _i = _perm.take(_F)
    _synapses_i = _synapses[_i]
    Apost[_synapses_i]=Apost[_synapses_i]*exp(-(-lastupdate[_synapses_i] + t)/taupost)
    Apre[_synapses_i]=Apre[_synapses_i]*exp(-(-lastupdate[_synapses_i] + t)/taupre)
    _target_ge[_u]+=w[_synapses_i]
    Apre[_synapses_i]+=dApre
    w[_synapses_i]=clip(w[_synapses_i]+Apost[_synapses_i],0,gmax)
    
    lastupdate[_synapses_i]=t
    
    _F += 1
    _F = extract(_flag.take(_F), _F)
    
Time taken: 289s versus 244s, i.e. extra takes 45s. Now previously Synapse.update
was taking 50% of the 260s, i.e. 130s, the first optimisation improved the speed
to 244s, so it was taking 114s. Around 10% of that is not the generated code, so
the generated code was taking about 102s. Doing it directly would take 45s, so
this should reduce the total by 102-45=57s, i.e. total time would be 187s. This
would then be [2x, 1.1x, 1.1x] slower than the Connection alternatives. In
other words, essentially the same speed when a like-for-like comparison is done,
suggesting that the C++ version should be as fast as the previous version.

Hacked C++ version:

Time taken 136s. This is [1.2, 0.7]x the time taken by the (C/Python) Connection
alternatives.
'''
from brian import *
from time import time
log_level_debug()

N = 1000
taum = 10 * ms
taupre = 20 * ms
taupost = taupre
Ee = 0 * mV
vt = -54 * mV
vr = -60 * mV
El = -74 * mV
taue = 5 * ms
F = 15 * Hz
gmax = .01
dApre = .01
dApost = -dApre * taupre / taupost * 1.05
dApost *= gmax
dApre *= gmax

eqs_neurons = '''
dv/dt=(ge*(Ee-vr)+El-v)/taum : volt   # the synaptic current is linearized
dge/dt=-ge/taue : 1
'''

if True:
    from scipy import weave
    class Synapses(Synapses):
        def update(self):
            if not hasattr(self, '_prepared_hack'):
                self._QNCV = []
                for queue, _namespace, code in zip(self.queues, self.namespaces, self.codes):
                    for _namespace in self.namespaces:
                        for k, v in _namespace.items():
                            if isinstance(v, float64):
                                _namespace[k] = float(v)
                    # we hack it to use C++, only works if the code below is
                    # updated when the Synapse equations change, and there is
                    # only one queue/namespace/code item.
                    orig_code_str = _namespace['_original_code_string']
                    code_str = None
                    if 'Apost[_synapses[_i]]=Apost[_synapses[_i]]*exp(-(-lastupdate[_synapses[_i]] + t)/taupost)' in orig_code_str:
                        code_str =  '''
                        for(int _spiking_synapse_idx=0;
                            _spiking_synapse_idx<_num_spiking_synapses;
                            _spiking_synapse_idx++)
                        {
                                const int _synapse_idx = _synapses[_spiking_synapse_idx];
                                const int _postsynaptic_idx = _post[_synapse_idx];

                                /* ORIGINAL CODE STRING WAS:
                                
                                Apost[_synapses[_i]]=Apost[_synapses[_i]]*exp(-(-lastupdate[_synapses[_i]] + t)/taupost)
                                Apre[_synapses[_i]]=Apre[_synapses[_i]]*exp(-(-lastupdate[_synapses[_i]] + t)/taupre)
                                _target_ge[_u]+=w[_synapses[_i]]
                                Apre[_synapses[_i]]+=dApre
                                w[_synapses[_i]]=clip(w[_synapses[_i]]+Apost[_synapses[_i]],0,gmax)
                                
                                lastupdate[_synapses[_i]]=t
                                
                                */                                
                                                                                            
                                double w_i = w[_synapse_idx];
                                if(lastupdate[_synapse_idx]<t) {
                                    Apost[_synapse_idx] *= exp(-(-lastupdate[_synapse_idx] + t)/taupost);
                                    Apre[_synapse_idx] *= exp(-(-lastupdate[_synapse_idx] + t)/taupre);
                                    lastupdate[_synapse_idx] = t;
                                }
                                _target_ge[_postsynaptic_idx] += w_i;
                                Apre[_synapse_idx] += dApre;
                                
                                w_i += Apost[_synapse_idx];
                                
                                if(w_i<0) w_i = 0;
                                else if(w_i>gmax) w_i = gmax;
                                w[_synapse_idx] = w_i;
                                                            
                        }
                        '''
                        vars = ['Apost', 'Apre', '_synapses', 't', 'taupost',
                                'taupre', '_target_ge', 'w', 'dApre', 'gmax',
                                '_post', 'lastupdate',
                                '_num_spiking_synapses']
                    elif 'Apost[_synapses]=Apost[_synapses]*exp(-(-lastupdate[_synapses] + t)/taupost)':
                        code_str =  '''
                        for(int _spiking_synapse_idx=0;
                            _spiking_synapse_idx<_num_spiking_synapses;
                            _spiking_synapse_idx++)
                        {
                                const int _synapse_idx = _synapses[_spiking_synapse_idx];
                                const int _postsynaptic_idx = _post[_synapse_idx];

                                /* ORIGINAL CODE STRING WAS:
                                
                                _post_neurons = _post[_synapses]
                                Apost[_synapses]=Apost[_synapses]*exp(-(-lastupdate[_synapses] + t)/taupost)
                                Apre[_synapses]=Apre[_synapses]*exp(-(-lastupdate[_synapses] + t)/taupre)
                                Apost[_synapses]+=dApost
                                w[_synapses]=clip(w[_synapses]+Apre[_synapses],0,gmax)
                                
                                lastupdate[_synapses]=t
                                
                                */
                                
                                double w_i = w[_synapse_idx];
                                if(lastupdate[_synapse_idx]<t) {
                                    Apost[_synapse_idx] *= exp(-(-lastupdate[_synapse_idx] + t)/taupost);
                                    Apre[_synapse_idx] *= exp(-(-lastupdate[_synapse_idx] + t)/taupre);
                                    lastupdate[_synapse_idx] = t;
                                }
                                Apost[_synapse_idx] += dApost;
                                
                                w_i += Apre[_synapse_idx];
                                
                                if(w_i<0) w_i = 0;
                                else if(w_i>gmax) w_i = gmax;
                                w[_synapse_idx] = w_i;
                                                            
                        }
                        '''
                        vars = ['Apost', 'Apre', '_synapses', 't', 'taupost',
                                'taupre', 'w', 'dApost', 'gmax',
                                '_post', 'lastupdate',
                                '_num_spiking_synapses']
                    else:
                        raise ValueError("Unknown code string")
                    self._QNCV.append((queue, _namespace, code_str, vars))                    
                self._prepared_hack = True
            if self._state_updater is not None:
                self._state_updater(self)
            for queue, _namespace, code_str, vars in self._QNCV:
                synaptic_events = queue.peek()
                if len(synaptic_events):
                    # Build the namespace - Here we don't consider static equations
                    _namespace['_synapses'] = synaptic_events
                    _namespace['t'] = float(self.clock._t)
                    _namespace['_num_spiking_synapses'] = len(synaptic_events)
                    weave.inline(code=code_str, arg_names=vars,
                                 local_dict=_namespace, compiler='gcc',
                                 extra_compile_args=['-O3', '-ffast-math',
                                                     '-march=native', # remove on old versions of gcc
                                                     ])
                queue.next()

input = PoissonGroup(N, rates=F)
neurons = NeuronGroup(1, model=eqs_neurons, threshold=vt, reset=vr)
S = Synapses(input, neurons,
             model='''w:1
                      dApre/dt=-Apre/taupre : 1 (event-driven)
                      dApost/dt=-Apost/taupost : 1 (event-driven)''',
             pre='''ge+=w
                    Apre+=dApre
                    w=clip(w+Apost,0,gmax)''',
             post='''Apost+=dApost
                     w=clip(w+Apre,0,gmax)''')
neurons.v = vr
S[:,:]=True
S.w='rand()*gmax'

def f():
    run(100 * second, report='text')

## do profiling
#import cProfile as profile
#import pstats
#profile.run('f()','stdp1_synapses.prof')
#stats = pstats.Stats('stdp1_synapses.prof')
##stats.strip_dirs()
#stats.sort_stats('cumulative', 'calls')
#stats.print_stats(50)

# do simple run
start = time()
f()
print 'Time taken:', time()-start

