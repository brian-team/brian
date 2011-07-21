from brian import *
from time import *
#Parameters
taum = 10*msecond
taue = 5*msecond
vt_input = -54*mvolt
vt_PC= -54*mvolt
vr = -60*mvolt
E1 = -60*mvolt
start_weight=0.6
tau_pre=20*ms
tau_post=tau_pre
gmax= 1
dA_pre=.01
dA_post=-dA_pre*tau_pre/tau_post*1.05
vr_input=vr # right?

#Signal1 + singal2 transformed
eqs_source1 = '''
Y1 = .5*sin(5*t/ms)*volt + .5*sin(7*t/ms)*volt : volt
dV/dt = -V/ms : 1
'''

#Signal1 + singal2 transformed
eqs_source2 = '''
Y2 = .5*sin(5*t/ms)*volt - .5*sin(7*t/ms)*volt : volt
dV/dt = -V/ms : 1
'''

#Rule for neuron encoding a source
eqs_inputs='''
dV/dt= (-(V-E1)+ s)/taum : volt
s : volt
'''

#Rule for neuron extracting the principal component
eqs_PC1_LIF='''
dV/dt = (-(V-E1)+ge)/taum : volt
dge/dt = -ge/taue : volt
'''

#Creating neurons
source1=NeuronGroup(1, model=eqs_source1) #'neuron' representing source1
source2=NeuronGroup(1, model=eqs_source2) #'neuron' representing source2
inputs=NeuronGroup(2, model=eqs_inputs, threshold=vt_input,reset=vr_input) #2 neurons, 1 for each source
inputs.V=vr
input1 = inputs.subgroup(1) #neuron converting source1 into a spike train
input2 = inputs.subgroup(1) #neuron converting souce2 into a spike train
PC=NeuronGroup(1, model=eqs_PC1_LIF, threshold=vt_PC, reset=vr)
#neuron for extracting the first principal component
PC.V=vr

#linking analogue value of source to a LIF neuron's voltage
input1.s=linked_var(source1, 'Y1')
input2.s=linked_var(source2, 'Y2')

#Excitatory connection between the inputs and the principal component
# extracting neuron, starting weight is set in the parameter section
Cmain = Connection(inputs, PC, 'ge', weight=start_weight*volt)

#Adding STDP to connection between inputs and principal component neuron
eqs_stdp='''
dA_pre/dt = -A_pre/tau_pre:1
dA_post/dt = -A_post/tau_post:1
'''

dA_post*=gmax
dA_pre*=gmax

#post code adapted to have synaptic scaling (sum of squaes of weights = 1)
stdp_with_scaling= STDP(Cmain, eqs_stdp, pre='A_pre+=dA_pre;w+=A_post',
                        post='A_post+=dA_post;w+=A_pre; w=w/(sum(w**2)**(.5))',
                        wmax=gmax)
#stdp_w/out_scaling= STDP(Cmain, eqs_stdp, pre='A_pre+=dA_pre;w
#+=A_post',post='A_post+=dA_post;w+=A_pre', wmax=gmax)

run(100*msecond)

plot(Cmain.W.todense(), '.')

print sum(Cmain.W.todense()**2)

xlim((-.5, 1.5))
show()
