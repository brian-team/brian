/*
This is an implementation of a benchmark described
in the following review paper:

Simulation of networks of spiking neurons: A review of tools and strategies (2006).
Brette, Rudolph, Carnevale, Hines, Beeman, Bower, Diesmann, Goodman, Harris, Zirpe,
Natschl�ger, Pecevski, Ermentrout, Djurfeldt, Lansner, Rochel, Vibert, Alvarez, Muller,
Davison, El Boustani and Destexhe.
Journal of Computational Neuroscience

Benchmark 2: random network of integrate-and-fire neurons with exponential synaptic currents

Clock-driven implementation with exact subthreshold integration
(but spike times are aligned to the grid)

R. Brette and Dan Goodman - May 2008
*/

#include <iostream>
#include<cstdio>
#include<utility>
#include<list>
#include<vector>
#include <time.h>

using namespace std;

#define taum 20.
#define taue 5.
#define taui 10.
#define Vt -50.
#define Vr -60.
#define El -49.
#define Nvals 1000, 2000, 4000, 8000, 16000, 32000
#define N_once 32000
#define N_varywe 4000
#define wevals 1.62, 2.4, 2.9, 3.2
#define Ne_prop 0.8
double we = (60*0.27/10.);
#define wi (-20*4.5/10.)
#define DURATION 2500.
#define dt .1
#define refrac 5.
#define REPEATS 10
#define BEST 7

const double Amatvals[3][3] = {
	{ 0.99501248,  0.,         0.         },
	{ 0.00493794,  0.98019867, 0.         },
	{ 0.00496265,  0.,         0.99004983 }
	};

const double Cvecvals[3] = { -2.44388520e-01, 0., 0. };

double Amat[3][3];
double Cvec[3];

bool use_connections=true;
bool vary_weights=false;
bool run_once=false;

void (*the_state_updater)(double &v, double &ge, double &gi);

void stateupdater_euler(double &v, double &ge, double &gi)
{// optimised this should come to 6 adds and 3 multiplies
	v+=(dt/taum)*(ge+gi-v+El); // optimised this would be v+=(dt/taum)*(ge+gi-v+El) where dt/taum is a const, so 1 mul and 4 adds
	ge+=(-dt/taue)*ge; // optimised this is 1 mul and 1 add
	gi+=(-dt/taui)*gi; // optimised this is 1 mul and 1 add
}

void stateupdater_matrix(double &v, double &ge, double &gi)
{// this comes to 9 muls, 9 adds and 3 copies
	static double vnew, genew, ginew;
	vnew = v*Amat[0][0]+ge*Amat[1][0]+gi*Amat[2][0]+Cvec[0];
	genew = v*Amat[0][1]+ge*Amat[1][1]+gi*Amat[2][1]+Cvec[1];
	ginew = v*Amat[0][2]+ge*Amat[1][2]+gi*Amat[2][2]+Cvec[2];
	v = vnew; ge = genew; gi = ginew; 
}

void stateupdater_matrix_efficient(double &v, double &ge, double &gi)
{
	// Note: 0s should be optimised away, and nonzeros should be replaced
	// with their constant values. Also note that v depends on initial
	// values of v, ge, gi but ge and gi only depend on ge, gi respectively
	// so we can do this calculation in this order without storing temporary
	// variables
	v = v*Amatvals[0][0]+ge*Amatvals[1][0]+gi*Amatvals[2][0]+Cvecvals[0];
	ge = v*Amatvals[0][1]+ge*Amatvals[1][1]+gi*Amatvals[2][1]+Cvecvals[1];
	gi = v*Amatvals[0][2]+ge*Amatvals[1][2]+gi*Amatvals[2][2]+Cvecvals[2];
}

pair<double,int> cuba(int N)
{
	int Ne = (int)((double)N*Ne_prop);
	int Ni = N-Ne;

	// State variables
	double v[N],ge[N],gi[N];
	for(int i=0;i<N;i++) {
		v[i]=Vr+(Vt-Vr)*double(rand())/(double)RAND_MAX;
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
	for(int i=0;i<N;i++)
		lastspike[i]=-1e10;
	if(use_connections)
	{
		for(int i=0;i<N;i++) {
			ntargets[i]=0;
			for(int j=0;j<N;j++)
				if (double(rand())/(double)RAND_MAX<80.0/(double)N) {
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
	}
	
	clock_t t1=clock();
	// Run the simulation
	int nspikes=0;
	for(double t=0.;t<DURATION;t+=dt) {
		for(int i=0;i<N;i++) {
			the_state_updater(v[i], ge[i], gi[i]);
			if (v[i]>Vt) {
				nspikes++;
				if(use_connections)
				{
					for(int j=0;j<ntargets[i];j++) {
						if (i<Ne)
							ge[targets[i][j]]+=weights[i][j];
						else
							gi[targets[i][j]]+=weights[i][j];
					}
				}
				v[i]=Vr;
				lastspike[i]=t;
			}
			if (t<=lastspike[i]+refrac)
				v[i] = Vr;
		}
	}

	clock_t t2 = clock();
	
	// Free things
	if(use_connections)
	{
		for(int i=0;i<N;i++) {
			delete [] targets[i];
			delete [] weights[i];
		}
	}
	
	return pair<double,int>((t2-t1)*1./CLOCKS_PER_SEC, nspikes);
}

bool compare_cubarun(pair<double, int> c1, pair<double, int> c2)
{
	return c1.first<c2.first;
}

pair<double,int> average_cuba(int N, int repeats, int best)
{
	list< pair<double,int> > cubaruns;
	for(int i=0;i<repeats;i++)
		cubaruns.push_back(cuba(N));
	cubaruns.sort(compare_cubarun);
	double t=0.0;
	int ns=0;
	int j=0;
	cout << "cur_result = []" << endl;
	for(list<pair<double,int> >::iterator i=cubaruns.begin();i!=cubaruns.end() && j<best;i++, j++)
	{
		t+=(*i).first;
		ns+=(*i).second;
		cout << "cur_result.append((" << (*i).first << ", " << (*i).second << "))" << endl;
	}
	return pair<double,int>(t/(double)best,ns/best);
}

void runs(vector<int> N, int repeats, int best)
{
	cout << "cpp_cuba = []" << endl;
	for(int i=0;i<N.size();i++)
	{
		pair<double, int> tn = average_cuba(N[i], repeats, best);
		cout << "cpp_cuba.append((" << N[i] << ", " << tn.first << ", " << tn.second << ", cur_result))" << endl;
	}
}

void runs_varywe(vector<double> wev, int repeats, int best)
{
	cout << "cpp_cuba = []" << endl;
	for(int i=0;i<wev.size();i++)
	{
		we = wev[i];
		pair<double, int> tn = average_cuba(N_varywe, repeats, best);
		cout << "# we = " << we << endl;
		cout << "cpp_cuba.append((" << N_varywe << ", " << tn.first << ", " << tn.second << ", cur_result))" << endl;
	}
}

int main(int argc, char *argv[]) {
	
	the_state_updater = stateupdater_euler;
	
	if(argc>1)
	{
		for(int i=1;i<argc;i++)
		{
			if(argv[i][0]=='h')
			{
				cout << "Command line options:" << endl << endl;
				cout << "s - no spiking" << endl;
				cout << "w - vary weights" << endl;
				cout << "m - state updater matrix" << endl;
				cout << "M - state updater efficient matrix" << endl;
				cout << "o - run once only" << endl;
				return 0;
			}
			if(argv[i][0]=='s')
			{
				cout << "# no spiking" << endl;
				use_connections = false;
			}
			if(argv[i][0]=='w')
			{
				cout << "# vary weights" << endl;
				vary_weights = true;
			}
			if(argv[i][0]=='m')
			{
				cout << "# state updater matrix" << endl;
				the_state_updater = stateupdater_matrix;
			}
			if(argv[i][0]=='M')
			{
				cout << "# state updater efficient matrix" << endl;
				the_state_updater = stateupdater_matrix_efficient;
			}
			if(argv[i][0]=='o')
			{
				cout << "# running once only" << endl;
				run_once = true;
			}
		}
	}
	
	srand((unsigned int)time(NULL));

	for(int i=0;i<3;i++)
	{
		Cvec[i] = Cvecvals[i];
		for(int j=0;j<3;j++)
			Amat[i][j]=Amatvals[i][j]; // do this to stop non-efficient matrix routine optimising it
	}
	
	int Nints[] = {Nvals};
	vector<int> N(Nints, Nints + sizeof(Nints) / sizeof(int) );
	
	double _we[] = {wevals};
	vector<double> wev(_we, _we + sizeof(_we) / sizeof(double));
	
	if(run_once)
	{
		pair<double,int> p = cuba(N_once);
		cout << "time_taken = " << p.first << endl;
		cout << "num_spikes = " << p.second << endl;
	} else {
		if(vary_weights)
			runs_varywe(wev, REPEATS, BEST);
		else
			runs(N, REPEATS, BEST);
	}
	
	return 0;
}

