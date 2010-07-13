#ifndef _BRIAN_LIB_H
#define _BRIAN_LIB_H

#include<vector>
#include<string>
#include<list>
#include<exception>
#include<stdexcept>

using namespace std;

#define neuron_value(group, neuron, state) group->S[neuron+state*(group->num_neurons)]

#define BrianException std::runtime_error

class CircularVector
{
private:
	inline int index(int i);
	inline int getitem(int i);
public:
	long *X, cursor, n;
	long *retarray;
	CircularVector(int n);
	~CircularVector();
	void reinit();
	void advance(int k);
	int __len__();
	int __getitem__(int i);
	void __setitem__(int i, int x);
	void __getslice__(long **ret, int *ret_n, int i, int j);
	void get_conditional(long **ret, int *ret_n, int i, int j, int min, int max, int offset=0);
	void __setslice__(int i, int j, long *x, int n);
	string __repr__();
	string __str__();
	void expand(long n);
};

class SpikeContainer
{
public:
	CircularVector *S;
	CircularVector *ind;
	int remaining_space;
	SpikeContainer(int m);
	~SpikeContainer();
	void reinit();
	void push(long *x, int n);
	void lastspikes(long **ret, int *ret_n);
	void __getitem__(long **ret, int *ret_n, int i);
	void get_spikes(long **ret, int *ret_n, int delay, int origin, int N);
	void __getslice__(long **ret, int *ret_n, int i, int j);
	string __repr__();
	string __str__();
};

#endif 
