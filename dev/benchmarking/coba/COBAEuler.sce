// COBAEuler
//
// This Scilab code is an implementation of a benchmark described
// in the following review paper:
//
// Simulation of networks of spiking neurons: A review of tools and strategies (2006).
// Brette, Rudolph, Carnevale, Hines, Beeman, Bower, Diesmann, Goodman, Harris, Zirpe,
// Natschläger, Pecevski, Ermentrout, Djurfeldt, Lansner, Rochel, Vibert, Alvarez, Muller,
// Davison, El Boustani and Destexhe.
// Journal of Computational Neuroscience
//
// Benchmark 1: random network of integrate-and-fire neurons with exponential synaptic conductances
//
// Clock-driven implementation with Euler integration
// (no spike time interpolation)
//
// The simulation takes about 100s on my PC
//
// ---------------------
//
// Romain Brette (June 2006)
// brette@di.ens.fr
//

// Parameters
Ne=3200; // Number of excitatory neurons
Ni=800; // Number of inhibitory neurons
N=Ne+Ni; // Total number of neurons
conProba=0.02; // Connection probability
taum=20; // Membrane time constant (ms)
taue=5; // Excitatory synaptic time constant
taui=10; // Inhibitory synaptic time constant
Vt = 10; // threshold, relative to rest (mV)
Vr = 0; // reset (mV)
Ee = 0+60; // excitatory reversal potential (relative to rest)
Ei = -80+60; // inhibitory reversal potential
dt=0.1; // time step
we=6/10; // excitatory synaptic weight (relative to gl)
wi=67/10; // inhibitory synaptic weight
duration=1000;  // duration of the simulation (ms)
refrac=5; // refractory period
stimulation=-1; // duration of the initial stimulation (<0: no stimulation)

// State variables (membrane potential, excitatory current, inhibitory current)
S=zeros(N,3);
if stimulation<0
  // No stimulation: the network must already be in an irregular state
  S(:,1)=grand(N,1,'nor',-5,5);
  S(:,2)=grand(N,1,'nor',4,1.5);
  S(:,3)=grand(N,1,'nor',20,12);
end

// Connectivity matrix: 2% connectivity with unitary weights
// This is a sparse matrix (otherwise it is too big and slow)
W=bool2s(sprand(N,N,conProba,'uniform')>0);

// Last spike times - for refractory period
LS=zeros(N,1)-1000;

// Simulation
printf("Starting simulation...\n");
timer();
t=0;
allspikes=[]; // Contains spikes (neuron,time)
while t<duration
  // STATE UPDATES (Euler)
  S(:,1)=S(:,1)+(dt/taum)*(-S(:,1)-S(:,2).*(S(:,1)-Ee)-S(:,3).*(S(:,1)-Ei));
  S(:,2)=S(:,2)*(1-dt/taue);
  S(:,3)=S(:,3)*(1-dt/taui);

  // Excitatory stimulation at 1000 Hz
  if t<stimulation
    S(:,2)=S(:,2)+we*grand(N,1,'poi',dt);
  end

  // Refractory period: membrane potential is clamped at reset
  S(find(LS>t-refrac),1)=Vr;

  // PROPAGATION OF SPIKES
  // Excitatory neurons
  spikes=find(S(1:Ne,1)>Vt); // List of neurons that meet threshold condition
  S(:,2)=S(:,2)+we*(sum(W(:,spikes),'c')); // Update the state of targets

  // Inhibitory neurons
  spikes=find(S(Ne+1:$,1)>Vt);
  S(:,3)=S(:,3)+wi*(sum(W(:,spikes),'c'));

  // Reset neurons after spiking
  spikes=find(S(:,1)>Vt);
  LS(spikes)=t; // Time of last spike
  S(spikes,1)=Vr; // Reset membrane potential
  
  // Store spike times
  allspikes=[allspikes;spikes',ones(spikes')*t]; // Each row is (neuron number,spike time)

  t=t+dt;
end
// Display the computation time
timer()

// Display the spikes
//clf; // Clear the display
//plot(allspikes(:,2),allspikes(:,1),'.');

