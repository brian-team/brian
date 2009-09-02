from brian import *
from modelfitting import *

#log_level_info()
#log_info('brian.library.modelfitting','Un message')

#dt          = .1*ms
#duration    = 200*ms
#data        = [(1,.1),(2,.3),(0,.15)]
#current     = 1.1*ones(int(duration/dt))
#eqs         = """
#                dv/dt = ((R*I-v)/tau) : 1
#                I : 1
#                R : 1
#                tau : second
#              """
#threshold   = 1
#reset       = 1

#params, value = modelfitting(eqs = eqs,
#                             reset = reset,
#                             threshold = threshold,
#                             data = data,
#                             input = current,
#                             dt = 0.1*ms,
#                             tau = (0*ms,5*ms,10*ms,30*ms),
#                             verbose = True
#                             )
#
#print 'Params', params
#print 'Value', value





def random_spike_trains(N1, N2, n):
    dt = defaultclock._dt
    
    target_train0 = cumsum(.005*rand(n)+.012)
    duration = target_train0[-1]+.005
    target_trains  = {}
    for i in range(N1):
        target_trains[i] = sort(floor((target_train0 + .001*(2*rand(n)-1))/dt)*dt)
#            target_trains[i][0] = 0
    
    model_trains  = {}
    for i in range(N2):
        model_trains[i] = sort(floor((target_train0 + .001*(2*rand(n)-1))/dt)*dt)
        model_trains[i][model_trains[i]<0] = 0
        model_trains[i] = sort(model_trains[i])
    return (target_trains, model_trains, duration)



def get_spiketimes(trains):
    spiketimes = []
    duration = 0.0
    for (i,train) in trains.iteritems():
        for t in train:
            spiketimes.append((i, t*second))
            if (t > duration):
                duration = t
    duration += .01
    spiketimes.sort(cmp = lambda x,y: int(x[1] < y[1]))
    return spiketimes


N = 3
(target_trains, model_trains, duration) = random_spike_trains(N, N, 20)
model_target = range(N)

print get_spiketimes(model_trains)


group   = SpikeGeneratorGroup(len(model_trains), get_spiketimes(model_trains))

cd = CoincidenceCounter(group, get_spiketimes(target_trains), model_target)
    

run(duration)

print cd.coincidences


