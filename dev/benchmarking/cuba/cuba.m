function [ttaken, nspikes] = cuba(N, use_connections, we)
% // CUBA
% //
% // This Matlab code is an implementation of a benchmark described
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
% // ---------------------
% //
% // Romain Brette (June 2006) and Dan Goodman (May 2008)
% // brette@di.ens.fr
% //
% 
% // Parameters
Ne=floor(N*0.8);
Ni=N-Ne;
conProba=80.0/N;
taum=20;
taue=5;
taui=10;
Vt = -50+49;
Vr = -60+49;
dt=0.1;
if nargin<3, we=60*0.27/10; end
wi=-20*4.5/10;
duration=2500;
refrac=5;

% Update matrix
A=[exp(-dt/taum),0,0;...
  we*taue/(taum-taue)*(exp(-dt/taum)-exp(-dt/taue)),exp(-dt/taue),0;...
  wi*taui/(taum-taui)*(exp(-dt/taum)-exp(-dt/taui)),0,exp(-dt/taui)];

%B=[exp(-dt/taum),0,0;...
%  taue/(taum-taue)*(exp(-dt/taum)-exp(-dt/taue)),exp(-dt/taue),0;...
%  taui/(taum-taui)*(exp(-dt/taum)-exp(-dt/taui)),0,exp(-dt/taui)]

% State variables (membrane potential, excitatory current, inhibitory current)
S=zeros(N,3);
S(:,1)=rand(N,1)*(Vt-Vr)+Vr; % Potential: uniform between reset and threshold

% Connectivity matrix: 80 synapses with unitary weights
% This is a sparse matrix (otherwise it is too big and slow)
if use_connections
    W=conmat(N,conProba);
end

% Last spike times - for refractory period
LS=zeros(N,1)-1000;

% Simulation
t1=clock;
t=0;
allspikes=[];
ns=0;
while t<duration
  S=S*A;

  S(LS>t-refrac,1)=Vr;

  spikesarr = S(:,1)>Vt;
  
  if use_connections
    spikes = spikesarr(1:Ne);
    S(:,2)=S(:,2)+(sum(W(:,spikes),2));
  
    spikes = spikesarr(Ne+1:end);
    S(:,3)=S(:,3)+(sum(W(:,spikes),2));
  end

  LS(spikesarr)=t;
  S(spikesarr,1)=Vr;
  ns=ns+sum(spikesarr);
  
  t=t+dt;
end
nspikes = ns;
ttaken = etime(clock,t1);
