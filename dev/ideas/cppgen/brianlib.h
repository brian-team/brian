#ifndef _BRIAN_LIB_H
#define _BRIAN_LIB_H

#include<vector>
#include<string>
#include<list>
using namespace std;

class StateUpdater;
class Threshold;
class Reset;

typedef list<int> SpikeList;

class NeuronGroup
{
public:
	double *S;
	int S_n, S_m;
	StateUpdater *su;
	Threshold *thr;
	Reset *reset;
	SpikeList last_spikes;
	NeuronGroup(double *S, int n, int m, StateUpdater *su, Threshold *thr, Reset *reset) :
		S(S), S_n(n), S_m(m), su(su), thr(thr), reset(reset) {}
	void update();
	void get_S_flat(double *S_out_flat, int nm);
};

class StateUpdater
{
public:
	virtual void __call__(NeuronGroup *group) {}; // SWIG can't handle pure virtual functions
};

class LinearStateUpdater : public StateUpdater
{
public:
	double *M;
	int M_n, M_m;
	double *b;
	int b_n;
	LinearStateUpdater(double *M, int M_n, int M_m, double *b, int b_n) :
		M(M), M_n(M_n), M_m(M_m), b(b), b_n(b_n) {}
	virtual void __call__(NeuronGroup *group); // like in Python
	string __repr__();
};

class Threshold
{
public:
	double value;
	int state;
	Threshold(int state, double value) : value(value), state(state) {}
	SpikeList __call__(NeuronGroup *group); // like in Python
};

class Reset
{
public:
	double value;
	int state;
	Reset(int state, double value) : value(value), state(state) {}
	void __call__(NeuronGroup *group); // like in Python
};

class NetworkOperation
{
public:
	virtual void __call__() {}
};

class StateMonitor : public NetworkOperation
{
public:
	NeuronGroup *group;
	int state;
	vector< vector<double> > values;
	StateMonitor(NeuronGroup *group, int state);
	virtual void __call__();
	vector<double> __getitem__(int i);
};

class Network
{
public:
	list<NeuronGroup*> groups;
	list<NetworkOperation*> operations;
	void add(NeuronGroup *group);
	void add(NetworkOperation *op);
	void update();
	void run(int timesteps);
};

#endif 
