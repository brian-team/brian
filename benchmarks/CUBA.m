% // CUBA
% //
% // This Scilab code is an implementation of a benchmark described
% // in the following review paper:
% //
% // Simulation of networks of spiking neurons: A review of tools and strategies (2006).
% // Brette, Rudolph, Carnevale, Hines, Beeman, Bower, Diesmann, Goodman, Harris, Zirpe,
% // Natschläger, Pecevski, Ermentrout, Djurfeldt, Lansner, Rochel, Vibert, Alvarez, Muller,
% // Davison, El Boustani and Destexhe.
% // Journal of Computational Neuroscience
% //
% // Benchmark 2: random network of integrate-and-fire neurons with exponential synaptic currents
% //
% // Clock-driven implementation with exact subthreshold integration
% // (but spike times are aligned to the grid)
% //
% // The simulation takes about 5s on my PC
% //
% // ---------------------
% //
% // Romain Brette (June 2006)
% // brette@di.ens.fr
% //
% 
% // Parameters
Ne=6400;%3200;
Ni=1600;%800;
N=Ne+Ni;
conProba=0.02;
taum=20;
taue=5;
taui=10;
Vt = -50+49;
Vr = -60+49;
dt=0.1;
we=60*0.27/10;
wi=-20*4.5/10;
duration=10000/4;
refrac=5;

% Update matrix
A=[exp(-dt/taum),0,0;...
  we*taue/(taum-taue)*(exp(-dt/taum)-exp(-dt/taue)),exp(-dt/taue),0;...
  wi*taui/(taum-taui)*(exp(-dt/taum)-exp(-dt/taui)),0,exp(-dt/taui)];

% State variables (membrane potential, excitatory current, inhibitory current)
S=zeros(N,3);
S(:,1)=rand(N,1)*(Vt-Vr)+Vr; % Potential: uniform between reset and threshold

% Connectivity matrix: 2% connectivity with unitary weights
% This is a sparse matrix (otherwise it is too big and slow)
W=sprand(N,N,conProba)>0;

% Last spike times - for refractory period
LS=zeros(N,1)-1000;

% Simulation
disp('Starting simulation...')
t1=clock;
t=0;
allspikes=[];
while t<duration
  S=S*A;

  S(find(LS>t-refrac),1)=Vr;

  spikes=find(S(1:Ne,1)>Vt);
  S(:,2)=S(:,2)+(sum(W(:,spikes),2));
  
  spikes=find(S(Ne+1:end,1)>Vt);
  % Actually there is a mistake: use spikes+Ne
  S(:,3)=S(:,3)+(sum(W(:,spikes),2));

  spikes=find(S(:,1)>Vt);
  LS(spikes)=t;
  S(spikes,1)=Vr;
  
  % allspikes=[allspikes;spikes,ones(size(spikes))*t];

  t=t+dt;
end
disp(etime(clock,t1))
