#!/usr/bin/env python
'''
Pierre Yger's winning entry for the 2012 Brian twister.
'''
from brian import *
import numpy, os, pylab

"""
An implementation of a simple topographical network, like those used in Mehring 2005 or Yger 2011. 
Cells are aranged randomly on a 2D plane and connected according to a gaussian profile
P(r) = exp(-d**2/(2*sigma**2)), with delays depending linearly on the distances. 
Note that the exact number of synapses per neuron is not fixed here. 

To avoid any border conditions, the plane is considered to be toroidal. 
Script will generate an Synchronous Irregular (SI) slow regime with propagating
waves that will spread in various directions, wandering over the network. 

In addition, an external layer of Poisson sources will stimulates some cells on the network, with
a wiring scheme such that the word BRIAN will pop up. External rates can be turned off to observed the 
spontaneous activity of the 2D layer. One can observe that despite the inputs is constant, the network
is not always responding to it. 

The script will display, while running, the spikes and Vm of the excitatory cells. 

Varying sigma will show the various activity structures from a random network (s_lat > 1), to a very
locally connected one (s_lat < 0.1)

"""

### We are setting the global timestep of the simulation
Clock(0.1 * ms)

### Cell parameters ###
tau_m   = 20. * ms # Membrane time constant
c_m     = 0.2 * nF # Capacitance
tau_exc =  3. * ms # Synaptic time constant (excitatory)
tau_inh =  7. * ms # Synaptic time constant (inhibitory)
tau_ref =  5. * ms # Refractory period
El      = -80 * mV # Leak potential
Ee      =  0. * mV # Reversal potential (excitation)
Ei      = -70.* mV # Reversal potential (inhibition)
Vt      = -50 * mV # Spike Threhold 
Vr      = -60 * mV # Spike Reset

### Equation for a Conductance-based IAF ####
eqs = Equations('''
dv/dt  = (El-v)/tau_m + (ge*(Ee-v)+gi*(Ei-v))/c_m : volt
dge/dt = -ge*(1./tau_exc) : uS
dgi/dt = -gi*(1./tau_inh) : uS
''')

n_cells       = 12500                   # Total number of cells
n_exc         = int(0.8 * n_cells)      # 4:1 ratio for exc/inh
size          = 1.                      # Size of the network
simtime       = 1000 * ms               # Simulation time
sim_step      = 1 * ms                  # Display snapshots every sim_step ms
epsilon       = 0.02                    # Probability density
s_lat         = 0.2                     # Spread of the lateral connections
g_exc         = 4.  * nS                # Excitatory conductance
g_inh         = 64. * nS                # Inhibitory conductance
g_ext         = 200 * nS                # External drive
velocity      = 0.3 * mm/ms             # velocity
ext_rate      = 100 * Hz                # Rate of the external source
max_distance  = size * mm/numpy.sqrt(2) # Since this is a torus
max_delay     = max_distance/velocity   # Needed for the connectors

### Generate the images with the letters B, R, I, A, N
### To do that, we create a png image and read it as a matrix
pylab.figure()
pylab.text(0.125, 0.4, "B R I A N", size=80)
pylab.setp(gca(), xticks=[], yticks=[])
pylab.savefig("BRIAN.png")
brian_letters = imread("BRIAN.png")
os.remove("BRIAN.png")
brian_letters = numpy.flipud(mean(brian_letters,2)).T
pylab.close()

### We create the cells and generate random positons in [0, size]x[0, size]
all_cells          = NeuronGroup(n_cells, model=eqs, threshold=Vt, reset=Vr, refractory=tau_ref)
all_cells.position = size*numpy.random.rand(n_cells, 2)
exc_cells          = all_cells[0:n_exc]
inh_cells          = all_cells[n_exc:n_cells]

### We initialize v values slightly above Vt, to have initial spikes
all_cells.v        = El + 1.1*numpy.random.rand(n_cells) * (Vt - El)

### Now we create the source that will write the word BRIAN
sources            = PoissonGroup(1, ext_rate)
sources.position   = array([[0.5, 0.5]])

### Function to get the distance between one position and an array of positions
### This is needed to used the vectorized form of the connections in the brian.Connection objects
### Note that the plane is wrapped, to avoid any border effects.
def get_distance(x, y):
    d1       = abs(x - y)
    min_d    = numpy.minimum(d1, size - d1)
    return numpy.sqrt(numpy.sum(min_d**2, 1))

### Function returning the probabilities of connections as a functions of distances
def probas(i, j, x, y):
    distance = get_distance(x[i], y[j])
    return epsilon * numpy.exp(-distance**2/(2*s_lat**2)) 

### Function returning linear delays as function of distances
def delays(i, j, x, y):
    distance = get_distance(x[i], y[j])
    return 0.1*ms + (distance * mm )/ velocity

### Function assessing if a cell is located in a particular letter of the word BRIAN
### Return 0 if not, and 1 if yes.
def is_in_brian(i, j, x, y):
    a, b         = brian_letters.shape
    tmp_x, tmp_y = (y[j][:, 0]*a).astype(int), (y[j][:, 1]*b).astype(int)
    return 1 - brian_letters[tmp_x, tmp_y]


print "Building network with wrapped 2D gaussian profiles..." 
Ce = Connection(exc_cells, all_cells, 'ge', weight=g_exc, max_delay=max_delay, 
            sparseness=lambda i, j : probas(i, j, exc_cells.position, all_cells.position), 
            delay     =lambda i, j : delays(i, j, exc_cells.position, all_cells.position))
Ci = Connection(inh_cells, all_cells, 'gi', weight=g_inh, max_delay=max_delay, 
            sparseness=lambda i, j : probas(i, j, inh_cells.position, all_cells.position), 
            delay     =lambda i, j : delays(i, j, inh_cells.position, all_cells.position))
Cext = Connection(sources, all_cells, 'ge', weight=g_ext, max_delay=max_delay, 
            sparseness=lambda i, j : is_in_brian(i, j, sources.position, all_cells.position))

print "--> mean probability from excitatory synapses:", Ce.W.getnnz()/float(n_exc*n_cells) * 100, "%"
print "--> mean probability from inhibitory synapses:", Ci.W.getnnz()/float((n_cells - n_exc)*n_cells) * 100, "%"


print "Setting the recorders..." 
v_exc   = RecentStateMonitor(exc_cells, 'v', record=True)
s_exc   = SpikeCounter(exc_cells)


ion() # To enter the interactive mode
print "Initializing the plots..."
fig1    = pylab.subplot(211)
im      = fig1.scatter(all_cells.position[:n_exc, 0], all_cells.position[:n_exc, 1], c=[0]*n_exc)
im.set_clim(0, 1)
fig1.set_ylabel("spikes")
pylab.colorbar(im) 
fig2    = pylab.subplot(212) 
im      = fig2.scatter(all_cells.position[:n_exc, 0], all_cells.position[:n_exc, 1], c=[0]*n_exc)
im.set_clim(El, Vt)
fig2.set_ylabel("v")
pylab.colorbar(im) 


manager = pylab.get_current_fig_manager() 

print "Running network ..." 
for time in xrange(int((simtime/sim_step)/ms)):
    run(sim_step)    
    fig1.cla()
    fig1.set_title("t = %g s" %((sim_step * time)/ms))   
    idx     = s_exc.count > 0
    if numpy.sum(idx) > 0:
        im = fig1.scatter(all_cells.position[:n_exc, 0][idx], all_cells.position[:n_exc, 1][idx], c=[0]*n_exc)
    s_exc.count = numpy.zeros(n_exc) ## We reset the spike counter
    fig1.set_xlim(0, size)
    fig1.set_ylim(0, size)
    fig1.set_ylabel("spikes")
    im.set_clim(0, 1)
    setp(fig1, xticks=[], yticks=[])
    
    fig2.cla() 
    im = fig2.scatter(all_cells.position[:n_exc, 0], all_cells.position[:n_exc, 1], c=v_exc.values[-1])
    fig2.set_xlim(0, size)
    fig2.set_ylim(0, size)
    fig2.set_ylabel("v")
    im.set_clim(El, Vt)
    setp(fig2, xticks=[], yticks=[])

    manager.canvas.draw()
    manager.canvas.flush_events()
ioff() # To leave the interactive mode    
    
