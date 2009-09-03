% // COBA
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
% // Benchmark 1: random network of integrate-and-fire neurons with exponential synaptic conductances
% //
% // Clock-driven implementation with Euler integration
% //
% // ---------------------
% //
% // Romain Brette (June 2006)
% // brette@di.ens.fr
% //
% // 8s
% // Parameters
Ne=3200;
Ni=800;
N=Ne+Ni;
conProba=0.02;
taum=20;
taue=5;
taui=10;
Vt = 10;
Vr = 0;
dt=0.1;
we=6/10;
wi=67/10;
duration=1000;
refrac=5;
Ee=0+60;
Ei=-80+60;

% State variables (membrane potential, excitatory current, inhibitory current)
S=zeros(N,3);
S(:,1)=randn(N,1)*5-5;
S(:,2)=randn(N,1)*1.5+4;
S(:,3)=randn(N,1)*12+20;
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
  S(:,1)=S(:,1)+(dt/taum)*(-S(:,1)-S(:,2).*(S(:,1)-Ee)-S(:,3).*(S(:,1)-Ei));
  S(:,2)=S(:,2)*(1-dt/taue);
  S(:,3)=S(:,3)*(1-dt/taui);

  S(find(LS>t-refrac),1)=Vr;

  spikes=find(S(1:Ne,1)>Vt);
  S(:,2)=S(:,2)+we*(sum(W(:,spikes),2));
  
  spikes=find(S(Ne+1:end,1)>Vt);
  % Actually there is a mistake: use spikes+Ne
  S(:,3)=S(:,3)+wi*(sum(W(:,spikes),2));

  spikes=find(S(:,1)>Vt);
  LS(spikes)=t;
  S(spikes,1)=Vr;
  
  allspikes=[allspikes;spikes,ones(size(spikes))*t];

  t=t+dt;
end
disp(etime(clock,t1))
