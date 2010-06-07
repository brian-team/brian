speedtrack_desc='Connection speed (spiking every dt, full connectivity)'

from brian import *
from tracking import *
import gc

num_neurons_set=[10, 20, 50, 100]
structure_set=['sparse', 'dense', 'dynamic']

def run_barrage():
    results=[]
    for n in num_neurons_set:
        for structure in structure_set:
            t=TrackSpeed(n, structure)
            results.append((str(n)+' ('+structure+')', track(t), (n, structure)))
            gc.collect()
    return (speedtrack_desc, results)

def plot_results(name, results):
    for structure in structure_set:
        allt=[t for desc, t, args in results if args[1]==structure]
        alln=[args[0] for desc, t, args in results if args[1]==structure]
        plot(alln, allt, label=structure)
    legend()
    title(name)


class TrackSpeed(object):
    def __init__(self, num_neurons, structure='sparse'):
        s=arange(num_neurons)
        class thr(Threshold):
            def __call__(self, P):
                return s
        G=NeuronGroup(num_neurons, 'V:1', threshold=thr())
        H=NeuronGroup(num_neurons, 'V:1')
        C=Connection(G, H, structure=structure)
        C.connect_full(G, H, weight=1)
        self.net=Network(G, H, C)
        self.net.prepare()

    def run(self):
        self.net.run(1*second)

if __name__=='__main__':
    name, results=run_barrage()
    print name
    for desc, t, args in results:
        print desc, ':', t
    plot_results(name, results)
    show()
