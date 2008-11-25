#ifndef _BRIAN_LIB_H
#define _BRIAN_LIB_H

#include<vector>
#include<string>
#include<list>
using namespace std;

#define neuron_value(group, neuron, state) group->S[neuron+state*(group->num_neurons)]

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

#endif 
