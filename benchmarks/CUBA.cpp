/*
This is an implementation of a benchmark described
in the following review paper:

Simulation of networks of spiking neurons: A review of tools and strategies (2006).
Brette, Rudolph, Carnevale, Hines, Beeman, Bower, Diesmann, Goodman, Harris, Zirpe,
Natschläger, Pecevski, Ermentrout, Djurfeldt, Lansner, Rochel, Vibert, Alvarez, Muller,
Davison, El Boustani and Destexhe.
Journal of Computational Neuroscience

Benchmark 2: random network of integrate-and-fire neurons with exponential synaptic currents

Clock-driven implementation with exact subthreshold integration
(but spike times are aligned to the grid)

You need the (free) GSL library (for the random numbers; alternatively,
replace then random generators with standard functions).

R. Brette - Jan 2008
*/

#include <iostream>
using namespace std;
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <time.h>

#define taum 20.
#define taue 5.
#define taui 10.
#define Vt -50.
#define Vr -60.
#define El -49.
#define Ne 3200
#define Ni 800
#define N (Ne+Ni)
#define we (60*0.27/10.)
#define wi (-20*4.5/10.)
#define dt .1
#define refrac 5.

int main() {
	cout << "CUBA model" << endl;
	
	// Initialization of random numbers
	const gsl_rng_type * T;
	gsl_rng_env_setup();
	T = gsl_rng_default;
	gsl_rng *rnd = gsl_rng_alloc(T);
	gsl_rng_set(rnd,1000);

	// State variables
	double v[N],ge[N],gi[N];
	for(int i=0;i<N;i++) {
		v[i]=El;
		ge[i]=0.;
		gi[i]=0.;
	}
	
	// Connection matrix 
	int *targets[N];
	int ntargets[N];
	double *weights[N];
	double bweights[N];
	int btargets[N];
	double lastspike[N];
	for(int i=0;i<N;i++) {
		lastspike[i]=-1e10;
		ntargets[i]=0;
		for(int j=0;j<N;j++)
			if (gsl_rng_uniform(rnd)<0.02) {
				btargets[ntargets[i]]=j;
				if (i<Ne)
					bweights[ntargets[i]]=we;
				else
					bweights[ntargets[i]]=wi;
				ntargets[i]++;
			}
		targets[i]=new int[ntargets[i]];
		weights[i]=new double[ntargets[i]];
		for(int j=0;j<ntargets[i];j++) {
			targets[i][j]=btargets[j];
			weights[i][j]=bweights[j];
		}
	}
	
	cout << "Running..." << endl;
	clock_t t1=clock();
	// Run the simulation
	int nspikes=0;
	for(double t=0.;t<1000.;t+=dt) {
		for(int i=0;i<N;i++) { // Euler
			if (t>lastspike[i]+refrac) {
				v[i]+=dt*((ge[i]+gi[i]-(v[i]-El))/taum);
				ge[i]+=-dt*ge[i]/taue;
				gi[i]+=-dt*gi[i]/taui;
				if (v[i]>Vt) {
					nspikes++;
					for(int j=0;j<ntargets[i];j++) {
						if (i<Ne)
							ge[targets[i][j]]+=weights[i][j];
						else
							gi[targets[i][j]]+=weights[i][j];
					}
					v[i]=Vr;
					lastspike[i]=t;
				}
			}
		}
	}

	cout << "Done in " << (clock()-t1)*1./CLOCKS_PER_SEC << " s" << endl;
	cout << nspikes << " spikes\n";
	
	// Free things
	for(int i=0;i<N;i++) {
		delete [] targets[i];
		delete [] weights[i];
	}
	
	return 0;
}

