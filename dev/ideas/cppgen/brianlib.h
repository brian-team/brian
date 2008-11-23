#ifndef _BRIAN_LIB_H
#define _BRIAN_LIB_H

#include<vector>
#include<string>
#include<list>
using namespace std;

class StateUpdater;
class Threshold;
class ResetBase;
class SpikeContainer;
class CircularVector;

typedef list<int> SpikeList;

class NeuronGroup
{
public:
	double *S;
	int num_vars, num_neurons;
	StateUpdater *su;
	Threshold *thr;
	ResetBase *resetobj;
	SpikeContainer *LS;
	NeuronGroup(double *S, int n, int m, StateUpdater *su,
			    Threshold *thr, ResetBase *reset,
			    int ls_n, int ls_m);
	~NeuronGroup();
	SpikeList get_spikes(int delay=0);
	void update();
	void reset();
	void get_S_flat(double *S_out_flat, int nm);
};

//inline double& neuron_value(NeuronGroup *group, int neuron, int state);

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

class ResetBase
{
public:
	virtual void __call__(NeuronGroup *group) {};
};

class Reset : public ResetBase
{
public:
	double value;
	int state;
	Reset(int state, double value) : value(value), state(state) {}
	virtual void __call__(NeuronGroup *group); // like in Python
};

// Note that this Refractoriness has to be initialised with an integer period rather
// than a float one. TODO: change this?
class Refractoriness : public Reset
{
public:
	int period;
	Refractoriness(int state, double value, int period) : Reset(state, value), period(period) {}
	virtual void __call__(NeuronGroup *group); // like in Python
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

class CircularVector
{
private:
	inline int index(int i);
	inline int getitem(int i);
public:
	int *X, cursor, n;
	CircularVector(int n);
	~CircularVector();
	void reinit();
	void advance(int k);
	int __len__();
	int __getitem__(int i);
	void __setitem__(int i, int x);
	list<int> __getslice__(int i, int j);
	list<int> get_conditional(int i, int j, int min, int max, int offset=0);
	void __setslice__(int i, int j, int *x, int n);
	void __setslice__(int i, int j, list<int> &x);
	string __repr__();
	string __str__();
};

class SpikeContainer
{
public:
	CircularVector *S, *ind;
	SpikeContainer(int n, int m);
	~SpikeContainer();
	void reinit();
	void push(int *x, int n);
	void push(list<int> &x);
	SpikeList lastspikes();
	SpikeList __getitem__(int i);
	SpikeList get_spikes(int delay, int origin, int N);
	SpikeList __getslice__(int i, int j);
	string __repr__();
	string __str__();
};

#endif 
