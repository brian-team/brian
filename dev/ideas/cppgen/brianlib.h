#ifndef _BRIAN_LIB_H
#define _BRIAN_LIB_H

#include<vector>
#include<string>
#include<list>
using namespace std;

#define neuron_value(group, neuron, state) (group)->S[((neuron)+(group)->origin)+(state)*((group)->S_m)]
//#define neuron_value(group, neuron, state) (group)->S[((neuron))+(state)*((group)->num_neurons)]
#define matrix_value(matrix, i, j, n, m) (matrix)[(i)*(m)+(j)]

class StateUpdater;
class Threshold;
class ResetBase;
class SpikeContainer;
class CircularVector;
class Connection;

class NeuronGroup
{
public:
	double *S;
	int num_vars, num_neurons, S_m;
	StateUpdater *su;
	Threshold *thr;
	ResetBase *resetobj;
	SpikeContainer *LS;
	int *spikesarray;
	NeuronGroup *owner;
	int origin;
	NeuronGroup(double *S, int n, int m, StateUpdater *su,
			    Threshold *thr, ResetBase *reset,
			    int ls_n, int ls_m);
	NeuronGroup(NeuronGroup *parent, int origin, int length);
	~NeuronGroup();
	void get_spikes(int **ret, int *ret_n, int delay=0);
	void update();
	void reset();
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
	void __call__(int **ret, int *ret_n, NeuronGroup *group);
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
	// Putting this virtual destructor in causes a memory leak, need to work out why
	//virtual ~StateMonitor() {};
	virtual void __call__();
	vector<double> __getitem__(int i);
};

class Network
{
public:
	list<NeuronGroup*> groups;
	list<NetworkOperation*> operations;
	list<Connection*> connections;
	void add(NeuronGroup *group);
	void add(NetworkOperation *op);
	void add(Connection *conn);
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
	int *retarray;
	CircularVector(int n);
	~CircularVector();
	void reinit();
	void advance(int k);
	int __len__();
	int __getitem__(int i);
	void __setitem__(int i, int x);
	void __getslice__(int **ret, int *ret_n, int i, int j);
	void get_conditional(int **ret, int *ret_n, int i, int j, int min, int max, int offset=0);
	void __setslice__(int i, int j, int *x, int n);
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
	void lastspikes(int **ret, int *ret_n);
	void __getitem__(int **ret, int *ret_n, int i);
	void get_spikes(int **ret, int *ret_n, int delay, int origin, int N);
	void __getslice__(int **ret, int *ret_n, int i, int j);
	string __repr__();
	string __str__();
};

class ConnectionMatrix
{
public:
    virtual void add_row(int i, double *b, int b_n) {};
    virtual void add_rows(int *x, int n, double *b, int b_n);
};

class DenseConnectionMatrix : public ConnectionMatrix
{
public:
	double *W;
	int n, m;
	DenseConnectionMatrix(double *S, int n, int m) : W(S), n(n), m(m) {};
	virtual void add_row(int i, double *b, int b_n);
};

class Connection
{
public:
	NeuronGroup *source, *target;
	ConnectionMatrix *connmat;
	int state, delay;
	Connection(NeuronGroup *source, NeuronGroup *target, ConnectionMatrix *connmat,
				int state=0, int delay=0) : source(source), target(target),
				connmat(connmat), state(state), delay(delay) {}
	virtual void propagate(int *x, int n);
	virtual void do_propagate();
};

#endif 
