if __name__ == '__main__':
    from brian import *
    from modelfitting import *

    model = '''
        dV/dt=(R*I-V)/tau : 1
        I : 1
        R : 1
        tau : second
    '''
    threshold = 1
    reset = 0

    input = loadtxt('current.txt')
    spikes0 = loadtxt('spikes.txt')
    spikes = []
    for i in xrange(2):
        spikes.extend([(i, spike * second + 5 * i * ms) for spike in spikes0])

    results = modelfitting(model=model, reset=reset, threshold=threshold,
                            data=spikes,
                            input=input, dt=.1 * ms,
                            max_cpu=4,
                            particles=1000, iterations=3, delta=2 * ms,
                            R=[1.0e9, 8.0e9],
                            tau=[10 * ms, 40 * ms],
                            _delays=[-10 * ms, 10 * ms])
    print_results(results)

